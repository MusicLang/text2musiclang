def get_amp_figure(velocity):
    """ """
    n = velocity / 120
    if n is None or n <= 0:
        return 'n'
    elif n <= 0.16:
        return 'ppp'
    elif n <= 0.26:
        return 'pp'
    elif n <= 0.36:
        return 'p'
    elif n <= 0.5:
        return 'mp'
    elif n <= 0.65:
        return 'mf'
    elif n <= 0.8:
        return 'f'
    elif n <= 0.9:
        return 'ff'
    else:
        return 'fff'

def amp_figure_to_velocity(amp_figure):
    if amp_figure == 'n':
        return 0
    elif amp_figure == 'ppp':
        return int(120 * 0.16)
    elif amp_figure == 'pp':
        return int(120 * 0.26)
    elif amp_figure == 'p':
        return int(120 * 0.36)
    elif amp_figure == 'mp':
        return int(120 * 0.5)
    elif amp_figure == 'mf':
        return int(120 * 0.65)
    elif amp_figure == 'f':
        return int(120 * 0.8)
    elif amp_figure == 'ff':
        return int(120 * 0.9)
    else:
        return int(120 * 0.95)
