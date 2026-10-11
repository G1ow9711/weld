"""Analytic per-child quadrature for input-only parent competition.

Every child keeps its endpoints, so short children and both gap sides survive.
No interpolation crosses a child boundary. Single full circles need a separate
phase-invariant distance representation and are explicitly unsupported here.
"""
import numpy as np

SCHEMA = 'ANALYTIC_CHILD_ARCLENGTH_EVENTS_V1'


def sample_segment(segment, count=33):
    points = np.asarray(segment['points'], dtype=float)
    if count < 3 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Invalid sampling input')
    fraction = np.linspace(0., 1., count)
    if segment['kind'] == 'line':
        return points[0] + fraction[:, None] * (points[-1] - points[0])
    if segment['kind'] != 'arc':
        raise ValueError('Unsupported segment kind')
    arc = segment['arc_parameters']
    sweep = float(arc['sweep_rad'])
    if arc.get('is_closed') or abs(sweep) >= 2*np.pi-1e-8:
        raise ValueError('Full circle requires phase-invariant representation')
    center = np.asarray(arc['center_mm'], float)
    normal = np.asarray(arc['plane_normal'], float)
    normal /= np.linalg.norm(normal)
    u = np.asarray(arc['start_direction'], float)
    u /= np.linalg.norm(u)
    radius = float(arc['radius_mm'])
    v = np.cross(normal, u)
    angle = fraction * sweep
    samples = center + radius * (np.cos(angle)[:, None]*u + np.sin(angle)[:, None]*v)
    direct = max(np.linalg.norm(samples[0]-points[0]), np.linalg.norm(samples[-1]-points[-1]))
    reverse = max(np.linalg.norm(samples[-1]-points[0]), np.linalg.norm(samples[0]-points[-1]))
    if min(direct, reverse) > 1e-4:
        raise ValueError('Arc parameters and delivered endpoints disagree')
    return samples if direct <= reverse else samples[::-1]


def parent_geometry(parent):
    segments = parent['segments']
    lengths = np.asarray([s['length_mm'] for s in segments], float)
    clouds = [sample_segment(s) for s in segments]
    # Analytic line/arc centroid, independent of vertex density and direction.
    centroids = []
    for s, cloud in zip(segments, clouds):
        if s['kind'] == 'line':
            centroids.append((cloud[0]+cloud[-1])/2)
        else:
            a = s['arc_parameters']; theta = float(a['sweep_rad'])
            u = np.asarray(a['start_direction'], float);u /= np.linalg.norm(u)
            n = np.asarray(a['plane_normal'], float);n /= np.linalg.norm(n)
            centroids.append(np.asarray(a['center_mm']) + a['radius_mm']*(np.sinc(theta/np.pi)*u + (1-np.cos(theta))/theta*np.cross(n,u)))
    return np.concatenate(clouds), np.asarray([c[[0,-1]] for c in clouds]), np.average(centroids, axis=0, weights=lengths)


def endpoint_correspondence(a, b, closed):
    if len(a) != len(b):
        return 0.
    variants = [b, b[::-1, ::-1]]
    return min(float(np.linalg.norm(a-np.roll(v, k, axis=0), axis=2).max())
               for v in variants for k in (range(len(b)) if closed else [0]))
