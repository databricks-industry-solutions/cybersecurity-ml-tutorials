# Databricks notebook source
import os
import json
import re

cfg={}
cfg["useremail"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
cfg["username"] = cfg["useremail"].split('@')[0]
cfg["username_sql_compatible"] = re.sub('\W', '_', cfg["username"])
cfg["db"] = f"cyber_ml_{cfg['username_sql_compatible']}"
cfg["data_path"] = f"/tmp/{cfg['username_sql_compatible']}/cyber_ml/"
cfg["download_path"] = "/tmp/cyber_ml"

if "getParam" not in vars():
  def getParam(param):
    assert param in cfg
    return cfg[param]

print(json.dumps(cfg, indent=2))

