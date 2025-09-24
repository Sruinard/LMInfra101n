from lminfra101n import models, configs

def test_repo():
    cfg = configs.load_config(env="dev")
    repo = models.RepositoryModel(cfg.model)
    repo._merge()
