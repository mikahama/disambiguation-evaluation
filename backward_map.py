# reverse the absolute integers on the master map
from fw_master_map import fw_map
from proposition_map import props

def func_to_string(func):
    import inspect
    return str(inspect.getsourcelines(func)[0])[2:-5].lstrip()

bw_map = {}

# basic bw_map
for k,v in fw_map.items():
    for kk,vv in v.items():
        bw_map[int(vv[1])] = (k,kk)

# add propositions
for k,v in props.items():
    bw_map[int(v)] = ("prop", func_to_string(k))


#
