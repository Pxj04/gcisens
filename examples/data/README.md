# Example data

| File | Origin | Licence |
|---|---|---|
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` | [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), a fictional sample dataset created by IBM data scientists for IBM Watson Analytics, published on Kaggle | Database: [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/); contents: [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) |

This repository redistributes the file unchanged (1,470 rows, 35 columns).
The examples and the article-reproduction tests read seven numeric columns (`Age`,
`DistanceFromHome`, `MonthlyIncome`, `NumCompaniesWorked`,
`PercentSalaryHike`, `TotalWorkingYears`, `YearsAtCompany`) and the
`Attrition` label.

The ODbL licence of the database applies to the data file only; the code of
`gcisens` stays under the MIT licence.

## File identity

The SHA-256 checksum of the CSV bytes in this repository is:

```text
e9f55fbf0a5c058306225d131311e135379d82ad0c94c33738ec75b9a179db9c
```

The article regression tests check this value before reading the data. Keep the
file unchanged for reproduction, including its line endings. Record a new
checksum and review the numerical results if you replace the dataset.
