# csdata

Single source of truth for dataset onboarding (download, clean, split, info.json
rendering, validation) across the synthetic-data project.

```python
import csdata
csdata.prepare("adult", out_dir="out/adult", schema="name", naming="real")
```

CLI: `csdata prepare adult --out out/adult --schema name --naming real`
