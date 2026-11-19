# Build the benchmark inputs from source data

These steps rebuild the histories and 20-candidate sets used for model
inference. They run on CPU. Run the commands from the repository root in a
Python 3.10 environment.

```bash
python -m pip install -r requirements-data.txt
mkdir -p data/raw/mind/MINDlarge/{MINDlarge_train,MINDlarge_dev}
mkdir -p data/raw/yelp
```

Raw files, Hugging Face caches, and processed bundles are local inputs.
Use the providers' access and redistribution terms when obtaining them.
See [data and model terms](../DATA_LICENSES.md).

## MIND-20

Download the **MINDlarge** training and validation archives from Microsoft
after accepting the research license. Save the archives as
`MINDlarge_train.zip` and `MINDlarge_dev.zip`, or adapt the paths in the commands
below. The labeled official development set provides the benchmark test
impressions. [Microsoft MIND download and format](https://msnews.github.io/#download)

```bash
unzip ~/Downloads/MINDlarge_train.zip -d data/raw/mind/MINDlarge/MINDlarge_train
unzip ~/Downloads/MINDlarge_dev.zip -d data/raw/mind/MINDlarge/MINDlarge_dev

python scripts/prepare_dataset.py --dataset mind
python scripts/build_mind_20.py
```

Each extracted directory must contain `news.tsv` and `behaviors.tsv`. The
loader uses the latest day in the training archive for validation and the
earlier training impressions for train-period popularity. Popularity counts
include clicked history entries and clicked candidates from those train
impressions. The loader retains histories of 10–50 known news items and caps
each initial split at 100,000 instances in source order. The second command
selects one clicked item and 19 distinct official impression negatives with a
seed derived from the impression ID, skips impressions with too few negatives,
and shuffles the 20 candidates. It writes `data/processed/mind_20.pkl` and
`results/mind_20_build_summary.json`.

## Amazon Books and Movies & TV

The downloader fetches the **rating-only 5-core benchmark CSV** and the **raw
item metadata JSONL** for each domain from the McAuley Lab dataset. Keep the
source order through conversion and preprocessing because candidate sampling
uses the ordered catalog. [Rating-only files](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark/5core/rating_only) ·
[Metadata files](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories)

| Domain | Ratings | Item metadata | Archived source rows |
| --- | --- | --- | --- |
| Books | `benchmark/5core/rating_only/Books.csv` | `raw/meta_categories/meta_Books.jsonl` | 9,488,297 ratings and 4,448,181 metadata rows |
| Movies & TV | `benchmark/5core/rating_only/Movies_and_TV.csv` | `raw/meta_categories/meta_Movies_and_TV.jsonl` | 7,441,129 ratings and 748,224 metadata rows |

```bash
python scripts/download_prepare_amazon_2023.py --dataset movies
python scripts/prepare_dataset.py --dataset amazon_movies

python scripts/download_prepare_amazon_2023.py --dataset books
python scripts/prepare_dataset.py --dataset amazon_books
```

The converter writes `reviews.parquet` and `meta.parquet` under
`data/raw/amazon/movies/` or `data/raw/amazon/books/`. Use `--cache_dir` to place
the Hugging Face downloads on a larger disk or `--revision` to select a Hub
revision. Re-running the downloader reuses existing Parquet files. Use
`--force` to rebuild them. The archived experiment has converted files and row
counts, and its download revision was not recorded. Use the reconstruction
check below to identify changes to the actual ranking inputs.

The loader joins ratings to metadata by `parent_asin`, keeps ratings ≥ 4 from
users with at least 12 eligible events, and orders by timestamp and item ID.
Each user's final two positive events supply the validation and test targets.
Earlier positives supply training popularity. Each history contains up to
50 events before its target, so test history can include the validation event.
Each ranking uses 19 distinct negatives sampled from
the target categories and then the full catalog, excluding that user's positive
items. Candidate text comes from metadata. The respective configs specify
which fields are shown and ablated. The loader processes the full Books catalog and interaction table in memory.
The provider lists the Books metadata download at approximately 15 GB.
Allow space for the Hub cache, converted Parquet, and processed bundle, and run
Books preprocessing on a high-memory node. [Books metadata file](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/raw/meta_categories/meta_Books.jsonl)

## Yelp Open Dataset

Download the Yelp Open Dataset archive through Yelp and follow its dataset
agreement. For an archive saved as `yelp_dataset.tar`, the extraction and loading
commands are below. [Yelp Open Dataset](https://www.yelp.com/dataset)

```bash
tar -xf ~/Downloads/yelp_dataset.tar -C data/raw/yelp
python scripts/prepare_dataset.py --dataset yelp
```

The loader expects these two files directly inside `data/raw/yelp/`. If the
archive includes a top-level directory, move the two extracted files to this
location.

```text
data/raw/yelp/yelp_academic_dataset_business.json
data/raw/yelp/yelp_academic_dataset_review.json
```

The loader keeps ratings ≥ 4 for businesses with metadata, sorts eligible
users' events by timestamp, business ID, and review ID, and applies the same
12-event minimum and leave-last-two-positive split as Amazon. To construct
item evidence, it scans review text again and reads snippets from training
review IDs in source order. Each business receives up to three nonempty
snippets of 240 characters each. Candidate fields are name, categories,
location, and the training-review summary. The result is
`data/processed/yelp.pkl`.

## Check the frozen subset and prompts

The committed manifest
[`aligned20_n1000_seed42.jsonl`](../configs/eval_subsets/aligned20_n1000_seed42.jsonl)
contains 1,000 test instances per dataset, shared by Llama and Qwen. The
manifest stores instance IDs and order. The dataset loaders and scenario
builders recover the candidate order, histories, and input text. Use the
committed selection for paper reproduction.

| Dataset | Processed bundle under `data/processed/` | Test pool | Selected | Generated Llama inputs |
| --- | --- | ---: | ---: | ---: |
| MIND-20 | `mind_20.pkl` | 57,550 | 1,000 | 11,000 |
| Amazon Books | `amazon_books.pkl` | 100,000 | 1,000 | 12,000 |
| Amazon Movies | `amazon_movies.pkl` | 100,000 | 1,000 | 10,163 |
| Yelp | `yelp.pkl` | 57,218 | 1,000 | 12,000 |

```bash
python scripts/check_reconstructed_inputs.py
# Check only one dataset as soon as its bundle is ready.
python scripts/check_reconstructed_inputs.py --dataset mind_20
```

The check reconstructs all scenario variants for the selected cases and
compares record counts and aggregate hashes of candidates, histories, and
prompt text against the archived paper inputs. It reports mismatches by dataset
and condition. The same check covers the Qwen Warm-start inputs and the
auxiliary Qwen control inputs, which were verified against the corresponding
Llama inputs. The public [fingerprints](../artifacts/metadata/input_reconstruction.json)
contain aggregate hashes and counts. Use `--processed_dir` to check bundles in
a custom output directory. Load pickle bundles only from preprocessing you
ran or from a trusted source.

To verify that seed 42 redraws the same selection from your reconstructed test
pools, build a copy of the manifest under `outputs/` and compare it with the
committed JSONL.

```bash
python scripts/build_eval_subset_manifest.py --subset_size 1000 --seed 42 \
  --output outputs/rebuilt-subset.jsonl
diff -u configs/eval_subsets/aligned20_n1000_seed42.jsonl outputs/rebuilt-subset.jsonl
```

Continue with [GPU ranking and UQ regeneration](GPU_REGENERATION.md).
