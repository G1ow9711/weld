"""Offline STEP contact-span diagnosis; reference geometry never becomes predictor input."""
import argparse
import json
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_local_curve_supervision_915 import compatible_interval
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    inv=read(OUT/'inventory.json')[21];folder=OUT/'source_21';freeze=read(folder/'input_freeze.json')
    assert digest(folder/'geometry.json')==freeze['geometry_sha256']
    assert digest(ROOT/inv['target']['target_path'])==inv['target']['target_sha256']
    assert digest(ROOT/inv['step']['STEP_path'])==inv['step']['STEP_sha256']
    geometry=read(folder/'geometry.json');target=read(ROOT/inv['target']['target_path']);child=target['parents'][0]['children'][0]
    reference_rep=child['world_representation'];gt=CurveCarrier(reference_rep);matches=[]
    for index,c in enumerate(geometry['contacts']):
        rep=dict(kind='line',endpoints_mm=[c['p0'],c['p1']]);event=compatible_interval(dict(representation=rep),reference_rep)
        if event is None:continue
        length=CurveCarrier(rep).length
        matches.append(dict(contact=index,materials=c['materials'],face_indices=c['face_indices'],
            length_mm=length,target_span_mm=event['span_mm'],
            start_margin_mm=event['lo'],end_margin_mm=length-event['hi'],
            maximum_geometric_deviation_mm=event['maximum_bidirectional_sample_error_mm']))
    cbf=ROOT/'DataSet/9月20日/GR768A/GR768A-0920-WQ_weld.cbf';raw=cbf.read_bytes().decode('utf-8',errors='replace')
    native,_=json.JSONDecoder().raw_decode(raw[raw.index('[\r\n'):]);matrix=native[0]['MatrixCollection'][0]
    fields={k:matrix[k] for k in ['HasStartWall','HasEndWall','FrontWallDistance','BackWallDistance','startBasePlateEdgeDistance','endBasePlateEdgeDistance']}
    sup=read(OUT/'run133/source_21_supervision.json');profiles=read(OUT/'run132/source_21/profiles.json')
    native_profiles=[dict(profile=int(k),length_mm=CurveCarrier(profiles[int(k)]['representation']).length,
        true_span_mm=v[0]['span_mm'],lo=v[0]['lo'],hi=v[0]['hi'],support_intervals=profiles[int(k)]['support_intervals_mm']) for k,v in sup['by_profile'].items()]
    result=dict(status='OFFLINE_REFERENCE_CONTACT_SPAN_DIAGNOSED',source=21,
        reference_geometry_sha256=digest(folder/'geometry.json'),STEP_sha256=inv['step']['STEP_sha256'],
        original_world_target_sha256=inv['target']['target_sha256'],native_cbf_sha256=digest(cbf),
        frame_provenance=inv['step']['verified_offline_frames'],
        offline_registration_GT_use_unknown=True,not_independent_geometry_certification=True,
        native_export_fields_are_not_new_customer_approved_E16_truth=True,native_export_fields=fields,
        STEP_contact_matches=matches,native_pure_XYZ_profiles=native_profiles,
        reference_face_count=len(geometry['faces']),reference_volume_count=len(set(f['volume'] for f in geometry['faces'])),
        native_plane_hypothesis_count=len(read(OUT/'run132/source_21/surfaces.json')),
        runtime_STEP_forbidden=True,no_fixed_margin_inserted=True,not_model_accuracy=True,full_contract_pass=False,end_utc=utc())
    write(dest/'receipt.json',result);print('REFERENCE_SPAN',matches,'EXPORT_FIELDS',fields,flush=True)

if __name__=='__main__':main()
