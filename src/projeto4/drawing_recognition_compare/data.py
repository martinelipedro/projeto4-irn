"""Duas apresentações do mesmo prefixo, sem rótulo fornecido à rede ou futuro."""
import numpy as np


def represent(points, max_events=180):
    points = np.asarray(points, np.float64)
    if not 1 <= len(points) <= max_events:
        raise ValueError('Prefixo deve ter entre 1 e 180 eventos.')
    low = points[:, :2].min(0)
    high = points[:, :2].max(0)
    extent = max(float((high - low).max()), 1.)
    xy = (points[:, :2] - (low + high) / 2) / extent
    delta = np.diff(xy, axis=0, prepend=xy[:1])
    dt = np.maximum(0, np.diff(points[:, 3], prepend=points[0, 3]))
    tokens = np.column_stack([delta, points[:, 2], np.log1p(dt / .1), xy]).astype(np.float32)
    padded = np.zeros((max_events, 6), np.float32)
    padded[:len(tokens)] = tokens
    pixel_xy = xy.astype(np.float64) * 27 + 15.5
    image = np.zeros((32, 32), np.uint8)
    def stamp(coords):
        ij = np.floor(coords + .5).astype(np.int64)
        for dx, dy in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
            xx, yy = ij[:, 0] + dx, ij[:, 1] + dy
            valid = (xx >= 0) & (xx < 32) & (yy >= 0) & (yy < 32)
            image[yy[valid], xx[valid]] = 1
    # A ordem de sobreposição é irrelevante para esta imagem binária.
    stamp(pixel_xy)
    coordinates = []
    for i in range(1, len(points)):
        if points[i-1, 2] >= .5:
            continue
        a, b = pixel_xy[i-1:i+1]
        n = max(1, int(np.ceil(np.max(np.abs(b-a)) * 2)))
        coordinates.append(a + (b-a) * (np.arange(n+1) / n)[:, None])
    if coordinates:
        stamp(np.concatenate(coordinates))
    return padded, image


def permute_strokes(points):
    """Inverte a ordem de traços inteiros; conserva pixels e geometria de cada traço."""
    points = np.array(points, np.float32, copy=True)
    stops = (np.flatnonzero(points[:, 2] > .5) + 1).tolist()
    if not stops or stops[-1] != len(points):
        stops.append(len(points))
    starts = [0] + stops[:-1]
    strokes = [points[a:b].copy() for a, b in zip(starts, stops) if b > a]
    if len(strokes) < 2:
        return points
    for stroke in strokes:
        stroke[-1, 2] = 1
    output = np.concatenate(strokes[::-1])
    # Relógio sintético monotônico: a imagem é idêntica; tempo é um controle OOD.
    output[:, 3] = np.arange(len(output), dtype=np.float32) * .1
    return output
