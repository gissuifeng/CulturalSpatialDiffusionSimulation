"""Tests use small synthetic fixtures, not scientific repetitions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from io import StringIO
import pipeline as p
np = p.np

def toy():
    genes=["N_B_0","H_B_0","N_C_0","H_C_0","R_0_0","I_B_C","A_B_C"]
    return dict(genes=genes,gi={g:i for i,g in enumerate(genes)},cultures=["B","C"],
        ids=["one","two"],native=np.array([0,1]),k=np.array([[1.,.7],[1.3,1.]]),
        a=np.array([0,0,1,1,-1,0,0]),b=np.array([-1,-1,-1,-1,-1,1,1]),
        x=np.array([[1000,0,0,101,10,13,7],[0,101,1000,0,10,13,7]],dtype=np.int64),
        out=[(np.array([1]),np.array([5]),5),(np.array([0]),np.array([5]),5)],
        carriers=np.array([[1,0],[1,0],[0,1],[0,1],[0,0],[1,1],[1,1]]),
        state=np.array([[1,0,0,0,0],[0,1,0,0,0],[1,0,0,0,0],[0,1,0,0,0],
                        [0,0,0,0,1],[0,0,1,0,0],[0,0,0,1,0]]))

class Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=p.read(p.HERE.parent/"config/design.json")
        self.rows=p.design(self.cfg)
        self.row=next(r for r in self.rows if r["is_baseline"])
    def test_design(self):
        self.assertEqual(len(self.rows),387)
        self.assertEqual(len({r["config_id"] for r in self.rows}),129)
        self.assertEqual(len({r["run_id"] for r in self.rows}),387)
        self.assertEqual(sum(r["family"]=="core" for r in self.rows),375)
        self.assertEqual(sum(r["is_baseline"] for r in self.rows),3)
        self.assertEqual(self.row["config_id"],"C068")
        for c in {r["config_id"] for r in self.rows}:
            self.assertEqual(len({r["seed"] for r in self.rows if r["config_id"]==c}),3)
    def test_bad_design(self):
        self.cfg["seeds"]=[1,1,2]
        with self.assertRaises(ValueError):p.design(self.cfg)
    def test_strength(self):
        x=p.transformed(toy(),dict(self.row,strength_a=0))
        np.testing.assert_array_equal(x["k"],np.ones((2,2)))
        y=p.transformed(toy(),dict(self.row,strength_a=2))
        np.testing.assert_allclose(y["k"],[[1,.4],[1.6,1]])
    def test_migration_matches_previous(self):
        inp=toy(); x=inp["x"].copy(); y=x.copy()
        p.engine.migrate(x,inp,1,self.row["seed"])
        flow,_,_,req=p.migrate_record(y,inp,1,self.row)
        np.testing.assert_array_equal(x,y)
        self.assertEqual(flow.sum(),req.sum())
    def test_scaled_migration_and_overdraw(self):
        inp=toy(); x=inp["x"].copy()
        flow,_,_,req=p.migrate_record(x,inp,1,dict(self.row,mobility_m=1.2))
        self.assertEqual(flow.sum(),12)
        self.assertEqual(x.sum(),inp["x"].sum())
        with self.assertRaises(ValueError):
            p.migrate_record(inp["x"].copy(),inp,1,dict(self.row,mobility_m=1000))
    def test_save_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(self.cfg,years=2);manifest=dict(settings=cfg,fingerprint="test")
            with patch.object(p,"checked_inputs",side_effect=lambda *args:toy()):
                self.assertEqual(p.simulate(self.row,tmp,manifest)[1],"completed")
                self.assertEqual(p.simulate(self.row,tmp,manifest)[1],"cached")
            folder=Path(tmp)/"runs"/self.row["run_id"]
            p.valid_run(folder,self.row,"test",deep=True)
            with np.load(folder/"states.npz") as z:
                self.assertEqual(z["counts"].shape,(3,2,7))
                self.assertEqual(z["after_migration"].shape,(2,2,7))
            with self.assertRaises(ValueError):
                p.valid_run(folder,dict(self.row,seed=123),"test")
            with self.assertRaises(ValueError):
                p.summarize(tmp,manifest,self.rows)
            (folder/"national_annual.csv").write_text("tampered")
            with self.assertRaises(ValueError):p.valid_run(folder,self.row,"test")
    def test_reproducible_seed(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            manifest=dict(settings=dict(self.cfg,years=2),fingerprint="test")
            with patch.object(p,"checked_inputs",side_effect=lambda *args:toy()):
                p.simulate(self.row,a,manifest);p.simulate(self.row,b,manifest)
            with np.load(Path(a)/"runs"/self.row["run_id"]/"states.npz") as x,np.load(Path(b)/"runs"/self.row["run_id"]/"states.npz") as y:
                np.testing.assert_array_equal(x["counts"],y["counts"])
    def test_progress_counts(self):
        progress=p.Progress(3);progress.tty=False
        with patch.object(p.sys,"stderr",new=StringIO()):
            progress.update("completed");progress.update("cached");progress.update("failed")
        self.assertEqual((progress.done,progress.completed,progress.cached,progress.failed),(3,1,1,1))

if __name__=="__main__":unittest.main(verbosity=2)
