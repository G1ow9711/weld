"""Independent geometry, coverage and metric checks for the supervised pipeline."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from training.labels import point_labels
from training.pilot import spatial_cores, scores


class PipelineTests(unittest.TestCase):
    def test_unicode_checkpoint_roundtrip(self):
        from training.scripts.train_pilot import atomic_save
        with tempfile.TemporaryDirectory(prefix='焊缝检查_') as folder:
            path=Path(folder)/'权重.pt'
            atomic_save({'x':torch.arange(9)},path)
            self.assertTrue(torch.equal(torch.load(path)['x'],torch.arange(9)))

    def test_finite_segment_distance_and_ignore_band(self):
        p=np.column_stack([np.linspace(0,10,21),np.zeros((21,2))])
        c={'samples':p,'sample_segment':np.zeros(len(p),dtype=np.int32),
           'starts':np.array([[0.,0,0]]),'ends':np.array([[10.,0,0]]),'instance':np.array([0])}
        q=np.array([[5,0,0],[5,3,0],[5,5,0],[5,7,0],[13,0,0],[20,0,0]],dtype=float)
        label,dist,inst,_=point_labels(q,c)
        np.testing.assert_allclose(dist,[0,3,5,7,3,10],atol=1e-6)
        np.testing.assert_array_equal(label,[1,1,-1,0,1,0])
        np.testing.assert_array_equal(inst,[0,0,-1,-1,0,-1])

    def test_whole_cloud_coverage_without_truth(self):
        xyz=np.random.default_rng(4).normal(size=(1003,3))
        xyz[:,2]=0
        leaves=list(spatial_cores(xyz,127))
        self.assertTrue(all(len(x)<=127 for x in leaves))
        np.testing.assert_array_equal(np.sort(np.concatenate(leaves)),np.arange(1003))

    def test_metric_counts_with_ignored_truth(self):
        m=scores(np.array([1,1,0,0,-1]),np.array([.9,.1,.9,.1,.9]))
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn'],m['ignored']),(1,1,1,1,1))
        self.assertAlmostEqual(m['precision'],.5)
        self.assertAlmostEqual(m['recall'],.5)
        self.assertAlmostEqual(m['f1'],.5)
        self.assertAlmostEqual(m['iou'],1/3)


if __name__=='__main__': unittest.main()
