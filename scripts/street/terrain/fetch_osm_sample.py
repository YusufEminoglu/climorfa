"""
Fetch OSM buildings with building:levels or height in Izmir FUA
Author: Yusuf Eminoglu / Antigravity
Date: 2026-10-29
"""

import os
import sys
import time
import json
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

def fetch_osm_buildings():
    out_path = "data/street/terrain/osm_buildings_fua.geojson"
    if os.path.exists(out_path):
        print(f"OSM dataset already exists at {out_path}")
        return gpd.read_file(out_path)
        
    print("Fetching OSM buildings across Izmir FUA urban cores...")
    boxes = [
        # (left, bottom, right, top) in WGS84
        (27.12, 38.41, 27.16, 38.44), # Konak / Alsancak
        (27.13, 38.44, 27.17, 38.47), # Alsancak / Port / Bayrakli
        (27.18, 38.44, 27.24, 38.48), # Bornova center & campus
        (27.08, 38.44, 27.13, 38.48), # Karsiyaka shore & center
        (27.00, 38.48, 27.06, 38.52), # Cigli / Egekent
        (27.14, 38.36, 27.19, 38.40), # Buca center & Tinaztepe
        (27.10, 38.37, 27.15, 38.41), # Karabaglar
        (27.13, 38.30, 27.18, 38.34), # Gaziemir
        (27.02, 38.38, 27.07, 38.41), # Balcova
        (26.96, 38.38, 27.01, 38.41), # Narlidere
        (26.86, 38.35, 26.92, 38.38), # Guzelbahce
    ]
    
    records = []
    for left, bot, right, top in boxes:
        url = f"https://api.openstreetmap.org/api/0.6/map?bbox={left},{bot},{right},{top}"
        req = urllib.request.Request(url, headers={"User-Agent": "IzmirResearch/1.0 (contact: yusuf.eminoglu@deu.edu.tr)"})
        try:
            print(f"Querying bbox ({left:.2f}, {bot:.2f}, {right:.2f}, {top:.2f})...", flush=True)
            with urllib.request.urlopen(req, timeout=25) as resp:
                root = ET.fromstring(resp.read())
                nodes = {n.attrib["id"]: (float(n.attrib["lon"]), float(n.attrib["lat"])) for n in root.findall("node")}
                for w in root.findall("way"):
                    tags = {t.attrib["k"]: t.attrib["v"] for t in w.findall("tag")}
                    if "building" in tags and ("building:levels" in tags or "height" in tags):
                        nd_refs = [nd.attrib["ref"] for nd in w.findall("nd")]
                        coords = [nodes[ref] for ref in nd_refs if ref in nodes]
                        if len(coords) >= 3:
                            # Parse levels
                            lvl_str = tags.get("building:levels", "")
                            lvl_val = None
                            try:
                                lvl_val = float(lvl_str.replace(",", ".").split(";")[0])
                            except:
                                pass
                                
                            # Parse height
                            h_str = tags.get("height", "")
                            h_val = None
                            try:
                                h_clean = h_str.replace("m", "").strip().replace(",", ".")
                                h_val = float(h_clean.split(";")[0])
                            except:
                                pass
                                
                            poly = Polygon(coords) if len(coords) >= 4 and coords[0] == coords[-1] else Point(sum(c[0] for c in coords)/len(coords), sum(c[1] for c in coords)/len(coords)).buffer(0.00005)
                            c_lon = sum(c[0] for c in coords) / len(coords)
                            c_lat = sum(c[1] for c in coords) / len(coords)
                            records.append({
                                "osm_id": w.attrib["id"],
                                "levels": lvl_val,
                                "height": h_val,
                                "raw_levels": tags.get("building:levels"),
                                "raw_height": tags.get("height"),
                                "lon": c_lon,
                                "lat": c_lat,
                                "geometry": poly
                            })
        except Exception as e:
            print(f"Bbox query skipped ({left:.2f}, {bot:.2f}): {e}")
        time.sleep(1)
        
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf = gdf.drop_duplicates(subset=["osm_id"]).reset_index(drop=True)
    # Project to EPSG:5253
    gdf_5253 = gdf.to_crs("EPSG:5253")
    gdf_5253.to_file(out_path, driver="GeoJSON")
    print(f"Successfully saved {len(gdf_5253):,} OSM buildings to {out_path}")
    return gdf_5253

if __name__ == "__main__":
    fetch_osm_buildings()
