import math


def vector_transform(value, mode='rectangle2polar'):
    if mode == 'rectangle2polar':
        x, y = value
        length = math.sqrt(x ** 2 + y ** 2)
        angle = math.atan2(y, x)
        output = [length, angle]
    elif mode == 'polar2rectangle':
        length, angle = value
        x = length * math.cos(angle)
        y = length * math.sin(angle)
        output = [x, y]
    else:
        raise(RuntimeError("invalid mode: ", mode))

    return output