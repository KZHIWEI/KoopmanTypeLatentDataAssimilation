from __future__ import annotations


def get(model: str):
    if model in ("NL-AE", "WL-AE", "KAE"):
        from src.eval.gaussianity.ensembles import variational

        return lambda ds, **kw: variational.run(ds, model, **kw)
    if model == "TensorVar":
        from src.eval.gaussianity.ensembles import closed_form

        return closed_form.run_tensorvar
    if model == "DMD":
        from src.eval.gaussianity.ensembles import closed_form

        return closed_form.run_dmd
    if model == "DBF":
        from src.eval.gaussianity.ensembles import filt

        return filt.run
    if model == "CGKN":
        from src.eval.gaussianity.ensembles import cgkn

        return cgkn.run
    raise KeyError(model)
