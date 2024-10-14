from symusic import Score, Track, TimeSignature, Tempo, Note
from .utils import amp_figure_to_velocity, get_amp_figure
import numpy as np
BASE_CHAR_ID = 33


class DecoderTokenizer:
    """
    Masked tokenizer for backing track generation
    Two use cases :

    1. Local Inpainting : Fill the mask for some bars
    2. Global Inpainting : Fill the mask for whole tracks
    3. Outpainting : The mask are located at the end of the sequence

    """
    SCORE_START = 'SCORE_START'
    UNKNOWN = 'UNKNOWN'
    CHORD_DURATION_NUM = 'CHORD_DURATION_NUM'
    CHORD_DURATION_DEN = 'CHORD_DURATION_DEN'
    CHORD_CHANGE = 'CHORD_CHANGE'
    CHORD_END = 'CHORD_END'
    MELODY_END = 'MELODY_END'
    WILL_END = 'WILL_END'
    DISSONANCE = 'DISSONANCE'
    AMPLITUDE = 'AMPLITUDE'
    INSTRUMENT_NAME = 'INSTRUMENT_NAME'
    INSTRUMENT_PART = 'INSTRUMENT_PART'
    DENSITY = 'DENSITY'
    AVERAGE_OCTAVE = 'AVERAGE_OCTAVE'
    NOTE_TYPE = 'NOTE_TYPE'
    NOTE_VAL = 'NOTE_VAL'
    NOTE_OCTAVE = 'NOTE_OCTAVE'
    NOTE_AMP = 'NOTE_AMP'
    NOTE_TIME = 'NOTE_TIME'
    NOTE_DURATION = 'NOTE_DURATION'
    NOTE_DURATION_NUM = 'NOTE_DURATION_NUM'
    NOTE_DURATION_DEN = 'NOTE_DURATION_DEN'
    PROMPT_START = 'PROMPT_START'
    PROMPT_END = 'PROMPT_END'
    END = 'END'
    MASKED = 'MASKED'
    BAR = 'BAR'
    END_OF_SONG = 'END_OF_SONG'

    CHORD_DESC_START = 'CHORD_DESC_START'
    CHORD_DESC_END = 'CHORD_DESC_END'
    CHORD_IDX = 'CHORD_IDX'
    UNSPECIFIED_CHORD = 'UNSPECIFIED_CHORD'

    TEMPO = 'TEMPO'
    CONTROL_MAX_POLYPHONY = 'CONTROL_MAX_POLYPHONY'
    CONTROL_MIN_POLYPHONY = 'CONTROL_MIN_POLYPHONY'
    SPACE = 'SPACE'


    def __init__(self,
                 tpq=24,  # Per bar
                 max_duration=8, # Nb quarters
                 max_voice=20,
                 max_bars=16,
                 min_bars=1,
                 local_masking_probability=0.3,
                 global_masking_probability=0.5,
                 proba_specified_chord=0.5,
                 proba_no_specified_chord=0.2,
                 proba_all_specified_chord=0.1,
                 max_polyphony_proba=0.3,
                 min_polyphony_proba=0.2,
                 max_max_polyphony=6,
                 max_min_polyphony=6,
                 density_proba=0.2,
                 special_notes_proba=0.2,
                 register_proba=0.2,
                 distribution=(0.3, 0.5, 0.2),
                 chord_in_prompt=False,
                 training_mode=True,
                 force_inference=False
                 ):
        """

        :param tpq: Ticks per quarter
        :param max_duration: Maximum duration of a note in quarters
        :param max_voice: Maximum number of voices
        :param max_bars: Maximum number of bars
        :param min_bars: Minimum number of bars
        :param local_masking_probability: Probability of masking a bar in local inpainting
        :param global_masking_probability: Probability of masking a track in global inpainting
        :param proba_specified_chord: Probability of having a specified chord in the prompt
        :param max_polyphony_proba: Probability of having a specified polyphony
        :param min_polyphony_proba: Probability of having a specified polyphony
        :param max_max_polyphony: Maximum number of maximum polyphony
        :param max_min_polyphony: Maximum number of minimum polyphony
        :param density_proba: Probability of having a specified density
        :param distribution: Distribution of the task chosen (local_inpainting_proba, global_inpainting_proba, outpainting_proba)
        :param chord_in_prompt: If the chord is in the prompt
        :param training_mode: If in training mode
        :param force_inference: If force inference

        """
        self.tpq = tpq
        max_duration = max_duration * tpq
        self.max_duration = max_duration
        self.max_voice = max_voice
        self.max_bars = max_bars
        self.min_bars = min_bars
        self.force_inference = force_inference
        self.training_mode = training_mode
        self.chord_in_prompt = chord_in_prompt
        self.local_masking_probability = local_masking_probability
        self.global_masking_probability = global_masking_probability
        self.distribution = distribution
        self.proba_specified_chord = proba_specified_chord
        self.max_quarters_time_signature = 8
        self._tokens_mapping = None
        self._inv_tokens_mapping = None
        self.max_polyphony_proba = max_polyphony_proba
        self.min_polyphony_proba = min_polyphony_proba
        self.density_proba = density_proba
        self.special_notes_proba = special_notes_proba
        self.register_proba = register_proba
        self.max_min_polyphony = max_min_polyphony
        self.max_max_polyphony = max_max_polyphony
        self.proba_no_specified_chord = proba_no_specified_chord
        self.proba_all_specified_chord = proba_all_specified_chord
        self.all_tokens = self.get_token_list()
        self.control_tokens = self.get_control_token_list()
        self.get_tokens_mapping()

    def get_token_list(self):

        def get_note_val_tokens():
            note_val_tokens = [self.NOTE_VAL + '__' + str(i) for i in range(12)]
            return note_val_tokens

        def get_note_octave_tokens():
            note_octave_tokens = [self.NOTE_OCTAVE + '__' + str(i) for i in range(0, 12)]
            return note_octave_tokens

        def get_note_amp_tokens():
            note_amp_tokens = [self.NOTE_AMP + '__' + amp for amp in ['n', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff']]
            return note_amp_tokens

        def get_note_duration_tokens():
            note_duration_tokens = [self.NOTE_DURATION + '__' + str(i) for i in range(0, self.max_duration + 1)]
            return note_duration_tokens

        def get_chord_duration_num_tokens():
            chord_duration_num_tokens = [self.CHORD_DURATION_NUM + '__' + str(i) for i in range(1, 17)]
            return chord_duration_num_tokens

        def get_chord_duration_den_tokens():
            chord_duration_den_tokens = [self.CHORD_DURATION_DEN + '__' + str(i) for i in [1, 2, 4, 8, 16]]
            return chord_duration_den_tokens

        def get_instrument_name_tokens():
            instrument_name_tokens = [self.INSTRUMENT_NAME + '__' + str(i) + "_0" for i in range(128)]
            instrument_name_tokens.append(self.INSTRUMENT_NAME + '__' + '0_1')
            return instrument_name_tokens

        def get_note_time_tokens():
            note_time_tokens = [self.NOTE_TIME + '__' + str(i) for i in range(0, (8 * self.tpq) + 1)]
            return note_time_tokens

        def get_mask_and_bars_tokens():
            bars = [self.BAR + '__' + str(i) for i in range(self.max_bars)]
            return [self.MASKED] + bars

        def get_chord_list_tokens():
            idxs = [self.CHORD_IDX + '__' + str(i) for i in range(self.max_bars)]
            return idxs
        
        def get_tempo_tokens():
            tempo_tokens = [self.TEMPO + '__' + str(i) for i in range(10, 300, 10)]
            return tempo_tokens

        all_tokens = [self.CHORD_CHANGE, self.MELODY_END, self.END, self.CHORD_END, self.END_OF_SONG]
        all_tokens += get_note_val_tokens()
        all_tokens += get_note_octave_tokens()
        all_tokens += get_note_amp_tokens()
        all_tokens += get_note_duration_tokens()
        all_tokens += get_chord_duration_num_tokens()
        all_tokens += get_chord_duration_den_tokens()
        all_tokens += get_instrument_name_tokens()
        all_tokens += get_note_time_tokens()
        all_tokens += get_mask_and_bars_tokens()
        all_tokens += get_chord_list_tokens()
        all_tokens += get_tempo_tokens()

        all_tokens = list(sorted(set(all_tokens)))

        return all_tokens


    def get_control_token_list(self):
        return [token for token in self.all_tokens if token.startswith('CONTROL')]
    

    def get_tokens_mapping(self):
        """
        Map all token idx to unique chars using ord
        :return:
        """
        all_tokens = self.get_token_list()
        tokens_mapping = {token: idx for idx, token in enumerate(all_tokens)}
        self._tokens_mapping = {token: chr(idx + BASE_CHAR_ID) for token, idx in tokens_mapping.items()}
        self._tokens_mapping[self.SPACE] = ' '
        self._inv_tokens_mapping = {char: token for token, char in self._tokens_mapping.items()}


    def tempo_to_tempo_token(self, tempo):
        return str(10 * int(tempo//10))

    def tokenize_bar(self, bar, idx_bar):
        """
        Tokenize a chord
        :param chord:
        :param idx_bar: int, index of the bar
        :param is_prompt: bool, if the chord is in the prompt of in the answer (change the tokenization)
        :param is_chord_prompt:  bool, if it is part of a chord progression prompt, in this case don't bother with chord time signature
        :return:
        """
        # roman, tonality_root, tonality_mode, chord_extension

        num, den, tempo = bar
        tokens = [
            self.CHORD_CHANGE,
            self.CHORD_IDX + '__' + str(idx_bar),
            self.CHORD_DURATION_NUM + '__' + str(num),
            self.CHORD_DURATION_DEN + '__' + str(den),
            self.TEMPO + '__' + str(self.tempo_to_tempo_token(tempo)),
            self.CHORD_END
        ]
        tokens = [t for t in tokens]
        return tokens



    def tokenize_track(self, program, is_drum):
        is_drum = int(is_drum)
        tokens = [
            self.INSTRUMENT_NAME + '__' + str(program) + '_' + str(is_drum),
        ]
        return tokens

    def tokenize_note(self, pitch, velocity, time, duration, is_drum):
        if velocity <= 2:
            return []
        
        val = pitch % 12
        oct = pitch // 12
        amp = get_amp_figure(velocity)
        duration = min(duration, self.max_duration)
        tokens = [
            self.NOTE_TIME + '__' + str(time),
            self.NOTE_VAL + '__' + str(val),
            self.NOTE_OCTAVE + '__' + str(oct),
            self.NOTE_AMP + '__' + amp,
        ]
        if not is_drum:
            tokens.append(self.NOTE_DURATION + '__' + str(duration))
        return tokens

    def add_to_notes(self, current_attributes, bar_start_time, tracks):
        current_attributes['note_pitch'] = 12 * current_attributes['note_octave'] + current_attributes['note_index']
        if current_attributes['note_velocity'] == 0:
            return
        note = Note(
            time=bar_start_time + int((current_attributes['note_time']) * 480 / self.tpq),
            duration=int(current_attributes['note_duration'] * 480 / self.tpq) if not current_attributes['is_drum'] else 100,
            pitch=current_attributes['note_pitch'],
            velocity=current_attributes['note_velocity'],
        )

        tracks[(current_attributes['program'], current_attributes['is_drum'], current_attributes['track_index'])].append(note)


    def untokenize(self, tokens):
        """
        Reconstruct the music data from the tokens
        :param tokens:
        :param tempo:
        :return:
        """
        # Initialize necessary structures for music data reconstruction
        notes = []  # This will store Note objects or similar structures
        bar_start_time = 0  # This will store the start time of the current bar
        next_bar_start_time = 0  # This will store the start time of the next bar
        current_attributes = {
            'program': 0,
            'is_drum': 0,
            'track_index': 0,
            'current_bar_idx': 0,
            'pitch': 0,
            'velocity': 'mf',
            'chord_duration_num': None,
            'chord_duration_den': None,
            'note_index': 0,
            'note_octave': 0,
            'note_velocity': 'mf',
            'note_pitch': None,  # This is a calculated attribute based on 'note_type' and 'note_index'
            'note_duration': None,
            'note_time': None,
            'current_tempo': 120,
        }

        tracks = {}
        track_keys = []
        current_time_signature = None
        time_signatures = []
        tempos = []
        current_tracks = {}
        will_stop = False
        # Iterate through the tokens to parse and reconstruct the music data
        for token in tokens:

            res = token.split('__')  # Split the token into key and value
            key = res[0]
            value = res[1] if len(res) > 1 else None
            # Parse token and update current attributes or create Note objects
            if key == self.CHORD_CHANGE:
                if will_stop:
                    print('Stopping...')
                    break
                # Handle chord change logic if necessary
                bar_start_time = next_bar_start_time
                current_tracks = {}
                pass
            elif key == self.CHORD_IDX:
                if will_stop:
                    print('Stopping...')
                    break
                current_attributes['current_bar_idx'] = int(value)
                if current_attributes['current_bar_idx'] >= self.max_bars - 1:
                    print(f"Reached the maximum number of bars: {self.max_bars}")
                    will_stop = True
                    break
            elif key == self.TEMPO:
                current_attributes['current_tempo'] = int(value)
                tempos.append(Tempo(bar_start_time, current_attributes['current_tempo']))
            elif key == self.CHORD_DURATION_NUM:
                current_attributes['chord_duration_num'] = int(value)
            elif key == self.CHORD_DURATION_DEN:
                current_attributes['chord_duration_den'] = int(value)
                next_bar_start_time = bar_start_time + int((4 * current_attributes['chord_duration_num'] / current_attributes['chord_duration_den']) * 480)
                new_time_signature = (current_attributes['chord_duration_num'], current_attributes['chord_duration_den'])
                if new_time_signature != current_time_signature:
                    # Handle time signature change logic if necessary
                    current_time_signature = new_time_signature
                    time_signatures.append(TimeSignature(
                        bar_start_time,
                        new_time_signature[0],
                        new_time_signature[1]
                    ))
            elif key == self.INSTRUMENT_NAME:
                program, is_drum = value.split('_')
                current_attributes['program'] = int(program)
                current_attributes['is_drum'] = bool(int(is_drum))
                program, is_drum = current_attributes['program'], current_attributes['is_drum']
                current_tracks[(program, is_drum)] = current_tracks.get((program, is_drum), 0) + 1
                current_attributes['track_index'] = current_tracks[(program, is_drum)]
                track_index = current_attributes['track_index']
                if (program, is_drum, track_index) not in tracks:
                    tracks[(program, is_drum, track_index)] = []
                    track_keys.append((program, is_drum, track_index))
            elif key == self.INSTRUMENT_PART:
                current_attributes['voice'] = int(value)
            elif key == self.NOTE_VAL:
                current_attributes['note_index'] = int(value)
            elif key == self.NOTE_OCTAVE:
                current_attributes['note_octave'] = int(value)
            elif key == self.NOTE_AMP:
                current_attributes['note_velocity'] = amp_figure_to_velocity(value)
                if current_attributes['is_drum']:
                    self.add_to_notes(current_attributes, bar_start_time, tracks)
            elif key == self.NOTE_TIME:
                current_attributes['note_time'] = int(value)
            elif key == self.NOTE_DURATION:
                current_attributes['note_duration'] = int(value)
                if not current_attributes['is_drum']:
                    self.add_to_notes(current_attributes, bar_start_time, tracks)
            elif key == self.MELODY_END:
                # Process end of melody token if necessary, for example, reset current attributes
                pass
        score = Score()
        score.tpq = 480
        score.time_signatures = time_signatures
        for tempo in tempos:
            score.tempos.append(tempo)

        for key in track_keys:
            notes = tracks[key]
            program, is_drum, track_index = key
            track = Track(program=program, is_drum=bool(is_drum))
            track.notes = list(sorted(notes, key=lambda x: x.time))
            if len(notes) == 0:
                self.add_default_track_events(track)
            score.tracks.append(track)
        return score


    def add_default_track_events(self, track):
        from symusic import ControlChange
        #track.controls = [ControlChange(0, 0, 0)]
        track.notes = [Note(0, 2, 60, 1)]


    def encode(self, midi_file, bar_range=None, remove_empty_bars=False):
        """
        Encode a midi file into a string of tokens
        :param midi_file:
        :param masked_bar_grid:
        :param chords:
        :param tags_grid: list[list][str], Tags for each bar
        :param bar_range:
        :param tempo:
        :return:
        """
        tokens = self.tokenize_for_inference(midi_file, bar_range=bar_range, remove_empty_bars=remove_empty_bars)
        chars = ''.join([self._tokens_mapping[t] for t in tokens ])
        return chars


    def decode(self, chars):
        chars = chars.replace('\t', '')
        chars = chars.replace("<pad>", "").replace("<s>", "").replace("<eos>", "")
        tokens = [self._inv_tokens_mapping.get(c, None) for c in chars]
        tokens = [t for t in tokens if t is not None]

        score = self.untokenize(tokens)
        return score

    def detokenize(self, tokens, tempo=120):
        tokens = [self._inv_tokens_mapping[c] for c in tokens]
        return tokens

    def partition(self, chords, tracks):
        """
        Partition the chords and tracks into chunks of 1 to max_bars (randomly)
        :param chords:
        :param tracks:
        :return:
        """
        len_chords = len(chords)
        partitions = []
        idx = 0
        while idx < len_chords:
            partition_length = np.random.randint(self.min_bars, self.max_bars + 1)
            partitions.append((idx, idx + partition_length))
            idx += partition_length

        # Get the effective partitions of chords and tracks
        partitions_chords = [chords[start:end] for start, end in partitions]
        partitions_tracks = [{k: v[start:end] for k, v in tracks.items()} for start, end in partitions]
        return partitions_chords, partitions_tracks

    def get_bar_grid(self, midi_file, bar_range=None, remove_empty_bars=True):
        from .parser import ParserMidi
        parser = ParserMidi(tpq=self.tpq)
        chords, tracks, track_keys = parser.parse(midi_file, bar_range=bar_range, remove_empty_bars=remove_empty_bars)
        if len(chords) > self.max_bars:
            raise ValueError(f"Too many bars in the midi file. Max bars allowed is {self.max_bars}")
        return chords, tracks, track_keys
    

    def tokenize_for_inference(self, midi_file, bar_range=None, remove_empty_bars=False):
        """
        Tokenize a midi file for inference
        :param midi_file:  str, Path to the midi file
        :param masked_bar_grid: np.array or list[list[int]], Masked bar grid size (n_tracks, n_bars)
        :param chords_prompt: list, List of chords (degree, tonality_degree, tonality_mode, chord_extension)
        :param tags_grid: list[list[list[str]]], Tags for each bar of each tracks (n_tracks, n_bars, <variable n tags>)
        :param bar_range: tuple, Range of chords to consider, if None, all the chords are considered
        :return:
        """
        bars, tracks, track_keys = self.get_bar_grid(midi_file, bar_range=bar_range, remove_empty_bars=remove_empty_bars)
        tokens = self._tokenize(bars, tracks, track_keys)
        return tokens

    def get_notes_onset_offsets(self, track_bar):
        start_times = []
        end_times = []
        for idx_note in range(len(track_bar['time'])):
            time = track_bar['time'][idx_note]
            duration = track_bar['duration'][idx_note]
            start_times.append(time)
            end_times.append(time + duration)
        return list(zip(start_times, end_times))


    def get_bar_duration(self, chord):
        num, den, chord_start, chord_end, tempo = chord
        return 4 * num / den

    def _tokenize(self, bars, tracks, track_keys):
        # Construct the prompt
        tokens = []
        for idx_bar, bar in enumerate(bars):
            tokens += self.tokenize_bar(bar, idx_bar)
            tokens += [self.SPACE]
            for idx_track, track_key in enumerate(track_keys):
                track = tracks[track_key]
                track_bar = track[idx_bar]
                _, program, is_drum, voice = track_key
                tokens += self.tokenize_track(program, is_drum)
                tokens += [self.SPACE]
                for idx_note in range(len(track_bar['time'])):
                    pitch = track_bar['pitch'][idx_note]
                    velocity = track_bar['velocity'][idx_note]
                    time = track_bar['time'][idx_note]
                    duration = track_bar['duration'][idx_note]
                    tokens += self.tokenize_note(pitch, velocity, time, duration, is_drum)  # Answer figure out the chord
                    tokens += [self.SPACE]
                tokens += [self.MELODY_END]

        tokens += [self.END]
        return tokens
    
    def _tokenize_without_prompt(self, tracks, track_keys):
        # Construct the prompt
        tokens = []
        nb_bars = len(tracks[track_keys[0]])
        
        # prompt_tokens = list(tokens)
        # Construct the answer, it does not matter to use chord or chord prompt here because if inference this is not used
        for idx_bar in range(nb_bars):
            tokens += self.tokenize_bar(None, idx_bar)
            for idx_track, track_key in enumerate(track_keys):
                track = tracks[track_key]
                track_bar = track[idx_bar]
                _, program, is_drum, voice = track_key
                tokens += self.tokenize_track(program, is_drum, voice)
                for idx_note in range(len(track_bar['time'])):
                    pitch = track_bar['pitch'][idx_note]
                    velocity = track_bar['velocity'][idx_note]
                    time = track_bar['time'][idx_note]
                    duration = track_bar['duration'][idx_note]
                    tokens += self.tokenize_note(pitch, velocity, time, duration, None, is_drum,
                                                    is_absolute=False)  # Answer figure out the chord
                tokens += [self.MELODY_END]

        tokens += [self.END]
        return tokens

