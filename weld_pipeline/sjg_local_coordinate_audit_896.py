"""Compare native workpiece-local exports with legacy PCD/world-target bindings."""
import argparse
import gc
import traceback
import numpy as np
import openpyxl
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_curve_label_inventory_895 import ROOT,OUT,read,write,digest,utc


def read_local_surface(path):
    header=[]
    with path.open('r',encoding='ascii') as stream:
        for line in stream:
            header.append(line.strip())
            if line.strip()=='end_header':break
        assert 'format ascii 1.0' in header
        assert any('unit millimeter; frame reference_workpiece' in x for x in header)
        count=int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
        xyz=np.loadtxt(stream,usecols=(0,1,2),dtype=np.float64,ndmin=2)
    assert xyz.shape==(count,3) and np.isfinite(xyz).all()
    return xyz,header


def workbook_labels(path):
    book=openpyxl.load_workbook(path,read_only=True,data_only=False);sheet=book['焊缝信息']
    rows=iter(sheet.iter_rows(values_only=True));header=next(rows)
    keys=[str(x or '').split('\n')[-1] for x in header];result=[]
    for number,row in enumerate(rows,2):
        d=dict(zip(keys,row));identity=d.get('ObjectId')
        if not identity:continue
        coords={}
        for key,field in [('local_start','StartMatrix'),('local_end','EndMatrix'),('source_start','Source StartMatrix'),('source_end','Source EndMatrix')]:
            value=[d.get(field+'.Translation.'+axis) for axis in 'XYZ']
            if all(type(v) in (int,float) and np.isfinite(v) for v in value):coords[key]=value
        result.append(dict(id=identity,row=number,coordinate_system=d.get('CoordinateSystem'),**coords))
    book.close();return result


def stats(values):
    a=np.asarray(values,float)
    return dict(count=int(a.size),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max())) if a.size else dict(count=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);inventory=read(OUT/'inventory.json');sources=[21,4,10,25,36,42,64];results=[]
    write(dest/'registered_protocol.json',dict(stage='NATIVE_LOCAL_VS_LEGACY_FRAME_AUDIT',sources=sources,
        user_requested_frame='workpiece_local',native_unit='mm',legacy_PCD_scale_to_mm=1000,
        read_only=True,no_optimizer_updates=True,no_model_scoring=True,no_GT_fitted_registration=True,
        source_sha256=digest(__file__),start_utc=utc()))
    try:
        for si in sources:
            raw=ROOT/inventory[si]['raw']['original_pcd_path'];folder=raw.parent
            books=list(folder.glob('*.xlsx'));surfaces=list(folder.glob('*_weld_workpiece.ply'));exports=list(folder.glob('*_weld_welds.json'))
            assert len(books)==len(surfaces)==1
            labels=workbook_labels(books[0]);local=np.array([r[k] for r in labels for k in ('local_start','local_end') if k in r])
            source=np.array([r[k] for r in labels for k in ('source_start','source_end') if k in r])
            native,header=read_local_surface(surfaces[0]);tree=cKDTree(native);old=read_xyz_pcd(raw)*1000.;oldtree=cKDTree(old)
            row=dict(source=si,name=inventory[si]['name'],native_file=str(surfaces[0].relative_to(ROOT)),native_sha256=digest(surfaces[0]),
                workbook=str(books[0].relative_to(ROOT)),workbook_sha256=digest(books[0]),legacy_file=str(raw.relative_to(ROOT)),legacy_sha256=digest(raw),
                native_points=len(native),legacy_points=len(old),native_header=header,native_bbox_mm=[native.min(0).tolist(),native.max(0).tolist()],
                legacy_bbox_mm=[old.min(0).tolist(),old.max(0).tolist()],
                local_labels_to_native_surface_mm=stats(tree.query(local,workers=2)[0]),
                local_labels_to_legacy_PCD_mm=stats(oldtree.query(local,workers=2)[0]),
                source_labels_to_legacy_PCD_mm=stats(oldtree.query(source,workers=2)[0]),
                source_labels_to_native_surface_mm=stats(tree.query(source,workers=2)[0]))
            target=read(ROOT/inventory[si]['target']['target_path']);parents={p['original_parent_id']:p for p in target['parents']};errors=[]
            for label in labels:
                if label['id'] not in parents or not all(k in label for k in ('source_start','source_end')):continue
                parent=parents[label['id']];a=np.asarray(parent['endpoints_mm']);b=np.asarray([label['source_start'],label['source_end']])
                errors.append(min(np.linalg.norm(a-b,axis=1).max(),np.linalg.norm(a-b[::-1],axis=1).max()))
            row['processed_parent_endpoints_vs_source_columns_mm']=stats(errors)
            if len(exports)==1:
                export=read(exports[0]);transform=np.asarray(export['exportToWorld']);rotation=transform[:3,:3];translation=transform[:3,3]
                assert export['coordinateFrame']=='reference_workpiece' and export['unit']=='millimeter'
                assert export['matrixConvention']=='row_major_column_vector' and np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7)
                errors=[]
                for label in labels:
                    for key in ('start','end'):
                        if 'local_'+key in label and 'source_'+key in label:errors.append(np.linalg.norm(rotation@label['local_'+key]+translation-label['source_'+key]))
                chosen=np.linspace(0,len(old)-1,min(20000,len(old)),dtype=np.int64);converted=(old[chosen]-translation)@rotation
                row.update(metadata_transform_file=str(exports[0].relative_to(ROOT)),metadata_transform_sha256=digest(exports[0]),
                    local_to_source_metadata_endpoint_error_mm=stats(errors),
                    legacy_cloud_to_native_after_metadata_mm=stats(tree.query(converted,workers=2)[0]),
                    export_to_world=transform.tolist())
            else:row['metadata_transform_status']='NO_NATIVE_JSON_TRANSFORM_NOT_GUESSED'
            write(dest/f'source_{si:02d}.json',row);results.append(row)
            print('FRAME_AUDIT',si,'LOCAL_NATIVE',row['local_labels_to_native_surface_mm'],'LOCAL_LEGACY',row['local_labels_to_legacy_PCD_mm'],
                'SOURCE_LEGACY',row['source_labels_to_legacy_PCD_mm'],flush=True)
            del native,tree,old,oldtree;gc.collect()
        write(dest/'receipt.json',dict(status='NATIVE_LOCAL_FRAME_AUDIT_COMPLETE',rows=results,
            geometric_surface_distances_are_not_weld_accuracy=True,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
