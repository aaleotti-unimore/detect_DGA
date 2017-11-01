import dask.bag as db
import json

def is_positive_filter(x):
    return x['label'] == 1
    pass


bag = db.read_text('tagged_dns_requests/*').map(json.loads)
bag = bag.map(is_positive_filter)
bag = bag.pluck('dns').pluck('rrname').distinct()
bag.to_textfiles('distinct_positive_domain_names/*')

