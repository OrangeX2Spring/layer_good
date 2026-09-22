"""Scene-mode correspondence cache with TUM's native interval phase."""
from kv_tracker.correspondence_cache import CorrespondenceCache
from kvt_tum_combined import CombinedSelector


class TUMCorrespondenceSelector(CombinedSelector):
    def __init__(self, config, log, inference_log):
        super().__init__(config, log, inference_log)
        self.cache_policy = CorrespondenceCache(config['patch_policy'],
            budget=config['cap'], interval=config['interval'], phase=1)
