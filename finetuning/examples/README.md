# Your decisions as a CSV

`decisions.csv` is a worked example: a 4-day week, 3 working hours a day, a late parcel, a discount
over Rs 500, team routing by repo and ticket urgency. Copy it, replace the rows with yours, then:

```sh
python finetuning/dataprep/import_csv.py yours.csv --name mydata --add-to-config
```

One row is one decision. **Put the fact the rule needs in the state, ask the question, give the right answer.**
Add near-miss rows around each threshold (499 / 500 / 501) so the model learns the line, not the topic.

| column | required | what | example |
|---|---|---|---|
| `state` | yes | the facts, as a JSON object or plain text | `{"order_total_rs": 501}` |
| `question` | yes | the instruction | `Does the order get the discount? Orders over Rs 500 get 10% off.` |
| `type` | no (noul) | `noul` (yes/no), `choice` or `score` | `choice` |
| `options` | choice, score | `a\|b\|c` or `a=what a means\|b=...`; score levels lowest first; noul: empty or `true=...\|false=...` | `payments=billing repos\|web=frontend repos` |
| `answer` | yes | noul: yes/no/true/false/1/0; choice: an option name; score: level name or index (0 = lowest) | `yes` |
| `split` | no | `train`, `test` or `gate`; empty = 90% train / 10% gate, fixed per row | `gate` |
| `source` | no (`your-data`) | a tag for where the row came from | `shop` |

JSON inside a CSV cell: wrap the cell in double quotes and double the quotes inside
(`"{""day"": ""Friday""}"`); any spreadsheet does this for you when you save as CSV.
The importer prints every bad row with its line number and how to fix it, and writes nothing until they are fixed
(or `--skip-bad`). Aim for 20+ rows per question, with no single answer above 80%.
