"""Freeze the TRAIN21-developed density decoder on the unchanged nine inputs."""
from weld_pipeline.sjg_local_runtime_904 import OUT,read
from weld_pipeline.sjg_native_instance_freeze_974 import main as freeze
from weld_pipeline.sjg_density_modes_978 import density_partition,controls


if __name__=='__main__':
    assert read(OUT/'run173/diagnostic_receipt.json')['status']=='TRAIN21_DENSITY_PARTITION_DIAGNOSTIC_COMPLETE'
    controls()
    freeze(partitioner=density_partition,method='density_modes')
