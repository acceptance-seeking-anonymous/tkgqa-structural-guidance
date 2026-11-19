import json, sys
new, ref = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
keys = ["n_examples", "list_sha256", "silver_sha256", "prompt_ids_sha256_first100", "lora", "opt", "max_len", "epochs", "total_steps", "n_silver", "len_mean", "chat_date", "prompt_sha256"]
res = {k: (new.get(k) == ref.get(k)) for k in keys}
ok = all(res.values()) and new.get("seed") != ref.get("seed") and not new.get("dry_run")
print(json.dumps(dict(STOP_RECIPE=not ok, checks=res, seed=new.get("seed"), versions=new.get("versions"), train_tok=new.get("train_tok"), grad_ckpt=new.get("grad_ckpt"))))
sys.exit(0 if ok else 4)
