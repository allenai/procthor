import json
import os
import sys
import inspect
import types
import copy
from multiprocessing import Pool, Value

from procthor.databases import DEFAULT_PROCTHOR_DATABASE

def delete_key_path(data, keys):
    if len(keys) == 1:
        if keys[0] in data:
            del data[keys[0]]
    else:
        if isinstance(data, list):
            for v in data:
                delete_key_path(data[keys[0]], keys[1:])
        elif isinstance(data, dict) and keys[0] in data:
            delete_key_path(data[keys[0]], keys[1:])

def get_key_path(data, keys):
    if not keys:
        return data
    else:
        if isinstance(data, list):
            out_list = []
            for indx in list(range(len(data))):
                v = data[indx]
                if keys[0] in v:
                    r = get_key_path(v[keys[0]], keys[1:])
                else:
                    r = None
                out_list.append(r)
            return out_list
        else:
            if keys[0] in data:
                return get_key_path(data[keys[0]], keys[1:])
            else:
                return None

def remap_keys(source, source_keys, root_out, out, keys, delete_source_key, key_depth=0, prev_out=None, all_keys=None):
    if len(keys) == 1:
        replace_val = get_key_path(source, source_keys)
        if delete_source_key:
            delete_key_path(root_out, source_keys)

        if replace_val:
            if isinstance(out, list) and isinstance(replace_val, list):
                for i in range(len(out)):
                    out[i][keys[0]] = replace_val[i]
            else:
                if not isinstance(out, dict):
                    prev_out[all_keys[-2]] = {}
                    out = prev_out[all_keys[-2]]
                out[keys[0]] = replace_val
    else:
        if isinstance(out, dict):
            remap_keys(source, source_keys, root_out, out[keys[0]], keys[1:], delete_source_key, key_depth + 1, out, keys)
        elif isinstance(out, list):
            new_source = get_key_path(source, source_keys[:key_depth] )
            for i in range(len(out)):
                remap_keys(new_source[i], source_keys[key_depth:], root_out, out[i], keys, delete_source_key, key_depth, out, keys)

class HouseVersionUpgrader(object):
    def upgrade_to(self, version, house):
        to_map = {
            (*([int(x) for x in n.split("__")[-1].split("_")]),):v for
                n, v in inspect.getmembers(self, inspect.ismethod)
                    if isinstance(v, types.MethodType) and n != 'upgrade_to'
        }

        if version not in to_map:
            raise ValueError(f"Invalid target version: '{version}'. Upgrade to valid version among: {', '.join([str(k) for k in to_map.keys()])}")
        return to_map[version](house)

class HouseUpgradeManager():
    @classmethod
    def parse_version(cls, version):
        return (*([int(v) for v in version.split("_")]),) if version is not None else (0, 0, 0)

    @classmethod
    def upgrade_to(cls, house, version):
        d = [c for c in dir(HouseUpgradeManager) if inspect.isclass(getattr(HouseUpgradeManager, c)) and getattr(HouseUpgradeManager, c)]
        from_map = {
            (*([int(x) for x in c.split("_")[1:]]),):
                getattr(HouseUpgradeManager, c)() for c in dir(HouseUpgradeManager)
            if inspect.isclass(getattr(HouseUpgradeManager, c)) and c != '__class__'
        }
        from_map[None] = from_map[(0, 0, 0)]

        house_version = HouseUpgradeManager.parse_version(house.get('version'))

        if house_version not in from_map:
            raise ValueError(f"Invalid source version: '{house_version}'. Upgrade from valid versions among: {', '.join([str(k) for k in from_map.keys()])}")

        if house_version == version:
            return house
        if house_version < version:
            return from_map[house_version].upgrade_to(version, house)
        else:
            raise ValueError(
                f"Invalid version: `{house_version}`. Must be lower than `{version}`")

    class From_0_0_0(HouseVersionUpgrader):
        def __1_0_0(self, house):
            out = copy.deepcopy(house)
            remapping = [
                (['proceduralParameters', 'ceilingMaterial'], ['proceduralParameters', 'ceilingMaterial', 'name'], False),
                (['proceduralParameters', 'ceilingMaterialTilingXDivisor'], ['proceduralParameters', 'ceilingMaterial', 'tilingDivisorX'], True),
                (['proceduralParameters', 'ceilingMaterialTilingYDivisor'], ['proceduralParameters', 'ceilingMaterial', 'tilingDivisorY'], True),
                (['rooms', 'floorMaterial'], ['rooms', 'floorMaterial', 'name'], False),
                (['rooms', 'floorMaterialTilingXDivisor'], ['rooms', 'floorMaterial', 'tilingDivisorX'], True),
                (['rooms', 'floorMaterialTilingYDivisor'], ['rooms', 'floorMaterial', 'tilingDivisorY'], True),
                (['objects', 'materialProperties'], ['objects', 'material'], True),
                # (['walls', 'materialProperties'], ['walls', 'material'], True),
                (['walls', 'materialId'], ['walls', 'material', 'name'], True),
                (['rooms', 'ceilings', 'materialProperties'], ['rooms', 'ceilings', 'material'], True),
                (['rooms', 'ceilings', 'material'], ['rooms', 'ceilings', 'material', 'name'], True),
                (['rooms', 'ceilings', 'tilingDivisorX'], ['rooms', 'ceilings', 'material', 'tilingDivisorX'], True),
                (['rooms', 'ceilings', 'tilingDivisorY'], ['rooms', 'ceilings', 'material', 'tilingDivisorY'], True),
                (['walls', 'materialProperties'], ['walls', 'material'], True),
                (['walls', 'material'], ['walls', 'material', 'name'], False),
                (['walls', 'color'], ['walls', 'material', 'color'], False),
            ]

            for (source_keys, target_keys, delete_source_key) in remapping:
                remap_keys(house, source_keys, out, out, target_keys, delete_source_key)

            hole_assets = DEFAULT_PROCTHOR_DATABASE.ASSET_ID_DATABASE

            for hole in out['windows'] + out['doors']:
                asset_id = hole['assetId']
                asset = hole_assets[asset_id]
                hole['holePolygon'] = [
                    hole['boundingBox']['min'],
                    hole['boundingBox']['max']
                ]

                offset = hole['assetOffset']

                hole['assetPosition'] = {
                    'x': hole['boundingBox']['min']['x'] + offset['x'] + asset["boundingBox"]["x"]/2.0,
                    'y': hole['boundingBox']['min']['y'] + offset['y'] + asset["boundingBox"]["y"]/2.0,
                    'z': 0
                }

                del hole['boundingBox']
                del hole['assetOffset']

            for wall in out['walls']:
                wall_id = wall['id']
                if wall_id.split('|')[1] == 'exterior':
                    wall['roomId'] = 'exterior'
            return out