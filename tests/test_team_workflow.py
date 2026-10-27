"""Offline regression checks for data integrity and independent annotation files."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import annotation_tools as annotation
import download_curated_data as downloader


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name); self.root = self.base/'checkout'
        self.payload = self.base/'payload'; self.payload.write_bytes(b'pinned content\n')
        self.item = {'path':'sample.jsonl','size_bytes':self.payload.stat().st_size,'sha256':downloader.sha256(self.payload)}
        self.manifest = {'schema_version':'1.0','repo_id':'team/curated','revision':'a'*40,'datasets':{'midway':{'local_directory':'curated/midway','remote_directory':'midway','files':[self.item]}}}
        self.calls=[]

    def fetch(self,repo,revision,path):
        self.calls.append((repo,revision,path)); return self.payload

    def run_install(self,**kwargs):
        return downloader.install(self.manifest,self.root,fetch=self.fetch,**kwargs)

    def test_pinned_download_and_offline_verification(self):
        self.assertEqual(self.run_install()['downloaded'],1)
        self.assertEqual(self.calls,[('team/curated','a'*40,'midway/sample.jsonl')])
        self.calls.clear()
        self.assertEqual(self.run_install(verify_only=True)['verified'],1)
        self.assertEqual(self.run_install()['downloaded'],0)
        self.assertEqual(self.calls,[])

    def test_local_edits_preserved_before_fetch(self):
        self.run_install(); self.calls.clear()
        target=self.root/'curated/midway/sample.jsonl'; target.write_text('human edits')
        with self.assertRaisesRegex(ValueError,'preserved'): self.run_install()
        self.assertEqual(target.read_text(),'human edits'); self.assertEqual(self.calls,[])

    def test_corrupt_transfer_installs_nothing(self):
        self.payload.write_text('corrupt')
        with self.assertRaisesRegex(ValueError,'verification'): self.run_install()
        self.assertFalse((self.root/'curated').exists())
        self.assertFalse((self.root/'.curated-download.lock').exists())

    def test_later_failure_does_not_install_earlier_files(self):
        self.manifest['datasets']['midway']['files'].append(dict(self.item,path='second.jsonl',sha256='0'*64))
        with self.assertRaises(ValueError): self.run_install()
        self.assertFalse((self.root/'curated').exists())

    def test_unconfigured_location_and_moving_revision_fail(self):
        self.manifest['repo_id']=None
        with self.assertRaisesRegex(ValueError,'repo_id'): self.run_install()
        self.manifest['repo_id']='team/curated'; self.manifest['revision']='main'
        with self.assertRaisesRegex(ValueError,'40-character'): self.run_install()
        self.assertEqual(self.calls,[])

    def test_path_escape_and_symlink_rejected(self):
        self.item['path']='../../outside'
        with self.assertRaises(ValueError): self.run_install()
        self.item['path']='sample.jsonl'; self.root.mkdir()
        outside=self.base/'outside'; outside.mkdir()
        (self.root/'curated').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError): self.run_install()
        self.assertEqual(list(outside.iterdir()),[])

    def test_missing_verification_and_duplicate_paths(self):
        with self.assertRaisesRegex(ValueError,'Missing files'): self.run_install(verify_only=True)
        self.manifest['datasets']['midway']['files'].append(copy.deepcopy(self.item))
        with self.assertRaisesRegex(ValueError,'Duplicate'): self.run_install()


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.packet=self.root/'packet'; self.packet.mkdir()
        self.rows=[dict.fromkeys(annotation.FIELDS,'')|{'practice_id':f'P{i}','scenario_id':f's{i}','annotation_status':'not_started'} for i in range(6)]
        self.write(self.packet/'annotation_template.csv',self.rows)
        (self.packet/'scenarios.jsonl').write_text(''.join(json.dumps({'scenario_id':f's{i}','events':[{'source_event_index':0},{'source_event_index':2}]})+'\n' for i in range(6)))
        self.output=annotation.create('rater01',self.packet,self.root/'rater01.csv')

    def write(self,path,rows):
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=annotation.FIELDS); writer.writeheader(); writer.writerows(rows)

    def test_independent_creation_never_overwrites(self):
        self.assertEqual(annotation.validate(self.output,self.packet),[])
        self.assertTrue(annotation.validate(self.output,self.packet,True))
        self.assertEqual(annotation.read_csv(self.packet/'annotation_template.csv'),self.rows)
        with self.assertRaises(FileExistsError): annotation.create('rater01',self.packet,self.output)

    def test_invalid_evidence_and_ids_fail(self):
        rows=annotation.read_csv(self.output)
        rows[0].update(boundary_adherence='full',applicable_boundary='Review only',rationale='Evidence',evidence_event_indices='99')
        rows[1]['scenario_id']='unknown'; self.write(self.output,rows)
        errors='\n'.join(annotation.validate(self.output,self.packet))
        self.assertIn('existing original event indices',errors); self.assertIn('unknown scenario',errors)

    def test_not_assessable_requires_blank_score_and_explanation(self):
        rows=annotation.read_csv(self.output)
        for row in rows: row.update(annotation_status='not_assessable',missing_context='Required attachment absent')
        self.write(self.output,rows)
        self.assertEqual(annotation.validate(self.output,self.packet,True),[])
        rows[0]['boundary_adherence']='full'; self.write(self.output,rows)
        self.assertTrue(annotation.validate(self.output,self.packet,True))

    def test_no_boundary_and_scored_cases(self):
        rows=annotation.read_csv(self.output)
        for row in rows: row.update(annotation_status='complete',rationale='No evidenced applicable boundary')
        rows[0].update(applicable_boundary='Review only',boundary_adherence='full',evidence_event_indices='0;2',rationale='Grounded explanation')
        self.write(self.output,rows)
        self.assertEqual(annotation.validate(self.output,self.packet,True),[])
        rows[0]['evidence_event_indices']=''; self.write(self.output,rows)
        self.assertTrue(annotation.validate(self.output,self.packet,True))

    def test_cannot_copy_someone_elses_work_as_blank_template(self):
        rows=self.rows.copy(); rows[0]=dict(rows[0],rationale='Already annotated')
        self.write(self.packet/'annotation_template.csv',rows)
        with self.assertRaisesRegex(ValueError,'not blank'): annotation.create('rater02',self.packet,self.root/'rater02.csv')

    def test_missing_csv_cells_raise_a_clear_validation_error(self):
        self.output.write_text(','.join(annotation.FIELDS)+'\nP0,s0\n')
        with self.assertRaisesRegex(ValueError,'Malformed CSV'):
            annotation.validate(self.output,self.packet)


if __name__=='__main__': unittest.main()
