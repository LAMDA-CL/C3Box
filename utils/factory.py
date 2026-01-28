def get_model(model_name, args):
    name = model_name.lower()
    if name=="proof":
        from models.proof import Learner
        return Learner(args)
    elif name == "simplecil":
        from models.simplecil import Learner
        return Learner(args)
    elif name =="zs_clip":
        from models.zs_clip import Learner
        return Learner(args)
    elif name == "coop":
        from models.coop_model import Learner
        return Learner(args)
    elif name == "l2p_without":
        from models.l2p_without import Learner
        return Learner(args)
    elif name == "rapf":
        from models.rapf import Learner
        return Learner(args)
    elif name == "coda":
        from models.coda_prompt import Learner
        return Learner(args)
    elif name == "dualprompt":
        from models.dual_prompt import Learner
        return Learner(args)
    elif name == "l2p":
        from models.l2p import Learner
        return Learner(args)
    elif name == "memo":
        from models.memo import Learner
        return Learner(args)
    elif name == "foster":
        from models.foster import Learner
        return Learner(args)
    elif name == "clg_cbm":
        from models.clg_cbm import Learner
        return Learner(args)
    elif name == "mind":
        from models.mind import Learner
        return Learner(args)
    elif name =="bofa":
        from models.bofa import Learner
        return Learner(args)
    elif name =="engine":
        from models.engine import Learner
        return Learner(args)
    elif name =="finetune":
        from models.finetune import Learner
        return Learner(args)
    elif name =="ease":
        from models.ease import Learner
        return Learner(args)
    elif name == "tuna":
        from models.tuna import Learner
        return Learner(args)
    elif name == "aper_adapter":
        from models.aper_adapter import Learner
        return Learner(args)
    elif name == "aper_ssf":
        from models.aper_ssf import Learner
        return Learner(args)
    elif name == "aper_vpt":
        from models.aper_vpt import Learner
        return Learner(args)
    elif name == "aper_finetune":
        from models.aper_finetune import Learner
        return Learner(args)
    else:
        assert 0
