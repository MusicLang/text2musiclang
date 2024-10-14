import symusic
import numpy as np
from fractions import Fraction as frac
from symusic import Track, TimeSignature, Note, Score, Tempo


class ParserMidi:

    def __init__(self, debug=False, separate_voices=False, tpq=24, min_dur=1):
        self.tpq = tpq
        self.min_dur = min_dur
        self.debug = debug
        self.separate_voices = separate_voices

    def get_bars(self, score):
        # Calculate bars
        time_signatures = score.time_signatures
        time_signatures = sorted(time_signatures, key=lambda x: x.time)
        if len(time_signatures) == 0:
            time_signatures = [TimeSignature(0, 4, 4)]
        ticks_per_quarter = score.ticks_per_quarter
        chord_durations = []
        time = 0
        bars = []
        max_note_onset = max([n.time for t in score.tracks for n in t.notes])
        while time <= max_note_onset:
            start_time = time
            end_time, num, den = self.get_end_of_bar(time, time_signatures, ticks_per_quarter)
            chord_durations.append((num, den))
            bars.append((start_time, end_time))
            time = end_time
        return bars, chord_durations

    def get_bar_nb_array_optimized(self, note_times, bars):
        # Convert bars to a NumPy array for efficient computation
        bars_array = np.array(bars)
        start_times = bars_array[:, 0]
        end_times = bars_array[:, 1]

        # Use broadcasting to find which bar each note belongs to
        # Note times are broadcast across the start and end times of bars to create a 2D boolean array
        in_bar = (note_times['time'][:, None] >= start_times) & (note_times['time'][:, None] < end_times)

        # Get the bar indices for each note. The result is a 2D array where each row has a single True value
        bar_indices = np.argmax(in_bar, axis=1)

        # Handle notes that do not fall within any bar by setting their bar index to a special value (e.g., -1)
        # This step is optional and depends on how you want to handle such cases
        outside_bars = ~in_bar.any(axis=1)
        bar_indices[outside_bars] = -1

        return bar_indices

    def get_end_of_bar(self, bar_start, time_signatures, ticks_per_quarter):
        # Find time signature
        # We use a tolrance of one quarter note to find the time signature
        res = [ts for ts in time_signatures if bar_start >= (ts.time - ticks_per_quarter//2)]
        time_signature = res[-1] if len(res) > 0 else TimeSignature(0, 4, 4)

        num = time_signature.numerator
        den = time_signature.denominator
        duration = 4 * frac(num, den)
        return int(bar_start + duration * ticks_per_quarter), num, den

    def groupby_bar(self, score_soa, bars):
        # Group SOA by bar
        bar_nb_array = score_soa['bar']
        # Time is relative to the start of the bar in our notation
        del score_soa['bar']
        result = [{k: score_soa[k][bar_nb_array == bar_nb] for k in score_soa.keys()} for bar_nb in range(len(bars))]
        return result

    def groupby_voices(self, tracks, program_voices_offset):
        new_tracks = {}
        for track_key, bars in tracks.items():
            # Determine the number of bars to ensure every voice has an entry for each bar
            num_bars = len(bars)

            # Initialize track_voices with an entry for each voice for each bar, even if empty
            track_voices = {}
            for bar_idx, bar in enumerate(bars):
                if 'voices' in bar:
                    unique_voices = np.unique(bar['voices'])
                else:
                    unique_voices = [0]  # Default voice if not specified

                for voice in unique_voices:
                    voice_key = (*track_key, voice)
                    if voice_key not in track_voices:
                        # Initialize with empty lists for each bar to ensure perfect mapping
                        track_voices[voice_key] = [{} for _ in range(num_bars)]

                    if 'voices' in bar:
                        # Select notes belonging to the current voice
                        mask = bar['voices'] == voice
                        bar_for_voice = {k: v[mask] for k, v in bar.items() if k != 'voices'}
                    else:
                        # If no voice separation, use the bar as is
                        bar_for_voice = bar

                    # Update the specific bar for the voice
                    track_voices[voice_key][bar_idx] = bar_for_voice

            # Update new_tracks with segregated bars
            new_tracks.update(track_voices)

        # Replace original tracks with newly segregated ones
        tracks.clear()
        tracks.update(new_tracks)

    def pprint(self, *text):
        if self.debug:
            print(*text)

    def preload_score(self, midi_file):
        score = symusic.Score(midi_file, ttype="tick")
        return score.resample(tpq=self.tpq, min_dur=self.min_dur)

    def find_time_bar(self, time, bars):
        for i, bar in enumerate(bars):
            if bar[0] <= time < bar[1]:
                return i
        return len(bars) - 1

    def average_tempo_grouped_by_bar(self, tempo, bars):
        tempo_bar = {}
        for t, bar_idx in tempo:
            if bar_idx not in tempo_bar:
                tempo_bar[bar_idx] = []
            tempo_bar[bar_idx].append(t)
        for bar_idx in tempo_bar.keys():
            tempo_bar[bar_idx] = sum(tempo_bar[bar_idx]) / len(tempo_bar[bar_idx])
        return tempo_bar
    

    def parse(self, midi_file, bar_range=None, remove_empty_bars=False):
        import time
        start = time.time()
        score_quantized = self.preload_score(midi_file)

        self.pprint("Resampling time: ", time.time() - start)
        start = time.time()
        bars, chord_durations = self.get_bars(score_quantized)
        self.pprint("Getting bars time: ", time.time() - start)

        tempos = [(t.qpm, t.time) for t in score_quantized.tempos] if len(score_quantized.tempos) > 0 else [(120, 0)]
        tempos = [(t[0], self.find_time_bar(t[1], bars)) for t in tempos]
        tempos = self.average_tempo_grouped_by_bar(tempos, bars)

        tracks = {}
        track_keys = []
        start = time.time()

        local_bar_range = (0, len(bars))
        program_voices_offset = {}

        for idx, track in enumerate(score_quantized.tracks):
            is_drum = track.is_drum
            program = track.program if not is_drum else 0
            score_soa = track.notes.numpy()
            track_keys.append((idx, program, is_drum, 0))

            start_time = bars[local_bar_range[0]][0]
            last_chord_index = min(local_bar_range[1], len(bars)) - 1
            end_time = bars[last_chord_index][1]
            mask = (score_soa['time'] >= start_time) & (score_soa['time'] < end_time)
            for k, v in score_soa.items():
                score_soa[k] = v[mask]

            if self.separate_voices:
                raise Exception("Separating voices is not implemented")

            score_soa['bar'] = self.get_bar_nb_array_optimized(score_soa, bars)
            score_soa['start_bar_tick'] = np.array([bars[bar_nb][0] for bar_nb in score_soa['bar']], dtype=np.int64)
            score_soa['time'] -= score_soa['start_bar_tick']
            grouped_score = self.groupby_bar(score_soa, bars)

            tracks[(idx, program, is_drum)] = grouped_score

        self.pprint("Grouping time: ", time.time() - start)
        start = time.time()

        chords = [list(c) for c in chord_durations]

        last_tempo = 120
        for i, chord in enumerate(chords):
            last_tempo = tempos.get(i, last_tempo)
            chord.append(last_tempo)

        self.pprint("Chord inference time: ", time.time() - start)
        start = time.time()
        self.groupby_voices(tracks, program_voices_offset)
        self.pprint("Voice groupby time: ", time.time() - start)

        if remove_empty_bars:
            chords, tracks, track_keys = self.remove_empty_bars_at_start_and_end(chords, tracks, track_keys)

        if bar_range is not None:
            tracks = {k: v[bar_range[0]:bar_range[1]] for k, v in tracks.items()}
            chords = chords[bar_range[0]:bar_range[1]]

        return chords, tracks, track_keys

    def remove_empty_bars_at_start_and_end(self, chords, tracks, track_keys):
        num_bars = len(chords)
        start_idx = 0
        end_idx = num_bars - 1

        # Function to check if a bar is empty
        def is_bar_empty(bar):
            if not bar:
                return True
            for arr in bar.values():
                if len(arr) > 0:
                    return False
            return True

        # Find the first non-empty bar index
        while start_idx <= end_idx:
            all_empty = True
            for track_key in tracks:
                bar = tracks[track_key][start_idx]
                if not is_bar_empty(bar):
                    all_empty = False
                    break
            if all_empty:
                start_idx += 1
            else:
                break

        # Find the last non-empty bar index
        while end_idx >= start_idx:
            all_empty = True
            for track_key in tracks:
                bar = tracks[track_key][end_idx]
                if not is_bar_empty(bar):
                    all_empty = False
                    break
            if all_empty:
                end_idx -= 1
            else:
                break

        # If all bars are empty, return empty structures
        if start_idx > end_idx:
            return [], {}, []

        # Slice chords and tracks to remove empty bars
        chords = chords[start_idx:end_idx + 1]
        for track_key in tracks:
            tracks[track_key] = tracks[track_key][start_idx:end_idx + 1]

        return chords, tracks, track_keys
    
