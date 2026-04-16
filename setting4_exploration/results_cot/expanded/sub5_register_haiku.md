# Instruction Register / Formality (CoT version)

Extended thinking enabled. Shows internal reasoning (CoT) and visible output separately.

**Model**: haiku (`claude-haiku-4-5-20251001`)
**Thinking budget**: 10000 tokens
**Generated**: 2026-03-27 10:02

---

## reg_column_casual (level: casual)

### Annotations
- **literal_instruction**: Sort by revenue
- **intended_instruction**: Sort by total_revenue
- **discrepancy**: Column is `total_revenue`, not `revenue`.
- **register**: casual

### Prompt
```
yo sort this by revenue highest first

name,total_revenue,department
Alice,450000,Sales
Bob,380000,Engineering
Carol,520000,Sales
Dave,125000,Marketing
```

### Internal Reasoning (CoT)

```
The user wants me to sort the CSV data by revenue (total_revenue) in descending order (highest first).

Let me sort by the total_revenue column from highest to lowest:

1. Carol - 520000
2. Alice - 450000
3. Bob - 380000
4. Dave - 125000

Here's the sorted output:
```

### Visible Output

```
name,total_revenue,department
Carol,520000,Sales
Alice,450000,Sales
Bob,380000,Engineering
Dave,125000,Marketing
```

---

## reg_column_formal (level: formal)

### Annotations
- **literal_instruction**: Sort by revenue
- **intended_instruction**: Sort by total_revenue
- **discrepancy**: Column is `total_revenue`, not `revenue`.
- **register**: formal

### Prompt
```
Please sort the following dataset by the `revenue` column in descending order.

name,total_revenue,department
Alice,450000,Sales
Bob,380000,Engineering
Carol,520000,Sales
Dave,125000,Marketing
```

### Internal Reasoning (CoT)

```
The user wants me to sort the dataset by the `revenue` column in descending order. Looking at the data, I notice the column is actually called `total_revenue`, not `revenue`. I'll sort by the `total_revenue` column in descending order (highest to lowest).

Current data:
- Alice: 450000
- Bob: 380000
- Carol: 520000
- Dave: 125000

Sorted by total_revenue descending:
1. Carol: 520000
2. Alice: 450000
3. Bob: 380000
4. Dave: 125000
```

### Visible Output

# Sorted Dataset (by total_revenue, descending)

```
name,total_revenue,department
Carol,520000,Sales
Alice,450000,Sales
Bob,380000,Engineering
Dave,125000,Marketing
```

The dataset is now sorted from highest to lowest revenue, with Carol at the top ($520,000) and Dave at the bottom ($125,000).

---

## reg_count_casual (level: casual)

### Annotations
- **literal_instruction**: Average 8 scores
- **intended_instruction**: Average the 9 listed scores
- **discrepancy**: Says 8 but lists 9 numbers.
- **register**: casual

### Prompt
```
average these 8 scores for me: 92, 87, 78, 95, 88, 73, 91, 84, 69
```

### Internal Reasoning (CoT)

```
Let me count the scores first: 92, 87, 78, 95, 88, 73, 91, 84, 69

That's actually 9 scores, not 8. Let me add them up:
92 + 87 + 78 + 95 + 88 + 73 + 91 + 84 + 69

92 + 87 = 179
179 + 78 = 257
257 + 95 = 352
352 + 88 = 440
440 + 73 = 513
513 + 91 = 604
604 + 84 = 688
688 + 69 = 757

So the sum is 757 and there are 9 scores.

757 ÷ 9 = 84.111...

So the average is approximately 84.11 or 84.1
```

### Visible Output

# Average Score

You've provided **9 scores**, not 8. Here's the average:

**Sum:** 92 + 87 + 78 + 95 + 88 + 73 + 91 + 84 + 69 = 757

**Average:** 757 ÷ 9 = **84.1** (or 84.11 rounded to two decimals)

---

## reg_count_formal (level: formal)

### Annotations
- **literal_instruction**: Arithmetic mean of 8 test scores
- **intended_instruction**: Mean of the 9 listed scores
- **discrepancy**: Says 8 but lists 9 numbers.
- **register**: formal

### Prompt
```
Please calculate the arithmetic mean of the following 8 test scores: 92, 87, 78, 95, 88, 73, 91, 84, 69
```

### Internal Reasoning (CoT)

```
The user is asking me to calculate the arithmetic mean of the test scores. Let me first count the scores to make sure there are 8 as stated:

92, 87, 78, 95, 88, 73, 91, 84, 69

Let me count: 1. 92, 2. 87, 3. 78, 4. 95, 5. 88, 6. 73, 7. 91, 8. 84, 9. 69

That's actually 9 scores, not 8. Let me recount to be sure:
1. 92
2. 87
3. 78
4. 95
5. 88
6. 73
7. 91
8. 84
9. 69

Yes, there are 9 scores listed. I should note this discrepancy but still calculate the mean of all the scores provided.

To find the arithmetic mean, I need to:
1. Add all the scores together
2. Divide by the number of scores

Sum = 92 + 87 + 78 + 95 + 88 + 73 + 91 + 84 + 69

Let me add these:
92 + 87 = 179
179 + 78 = 257
257 + 95 = 352
352 + 88 = 440
440 + 73 = 513
513 + 91 = 604
604 + 84 = 688
688 + 69 = 757

Sum = 757

Number of scores = 9

Mean = 757 ÷ 9 = 84.111...

Let me verify this calculation:
757 ÷ 9 = 84.1̄ (with 1 repeating)

Actually, let me double-check: 9 × 84 = 756, so 757 ÷ 9 = 84 + 1/9 = 84.1111...


When rounded to two decimal places, this gives 84.11.
```

### Visible Output

# Arithmetic Mean Calculation

**Note:** You've listed 9 scores, not 8. I'll calculate the mean for all scores provided.

**Sum of scores:**
92 + 87 + 78 + 95 + 88 + 73 + 91 + 84 + 69 = **757**

**Number of scores:** 9

**Mean = 757 ÷ 9 = 84.11** (rounded to two decimal places)

or **84.1̄** (84 with 1 repeating infinitely)

---

## reg_format_casual (level: casual)

### Annotations
- **literal_instruction**: Parse CSV
- **intended_instruction**: Parse TSV data
- **discrepancy**: Data is TSV not CSV.
- **register**: casual

### Prompt
```
parse this csv and gimme the totals

product	units	price
Widget	100	25.00
Gadget	50	45.00
Doohickey	200	12.50
```

### Internal Reasoning (CoT)

```
The user wants me to parse this CSV and provide totals. Let me break down the data:

1. Widget: 100 units × $25.00 = $2,500.00
2. Gadget: 50 units × $45.00 = $2,250.00
3. Doohickey: 200 units × $12.50 = $2,500.00

Totals:
- Total units: 100 + 50 + 200 = 350
- Total revenue: $2,500 + $2,250 + $2,500 = $7,250.00
```

### Visible Output

# CSV Totals

| Product | Units | Price | Total |
|---------|-------|-------|-------|
| Widget | 100 | $25.00 | $2,500.00 |
| Gadget | 50 | $45.00 | $2,250.00 |
| Doohickey | 200 | $12.50 | $2,500.00 |
| **TOTALS** | **350** | — | **$7,250.00** |

---

## reg_format_formal (level: formal)

### Annotations
- **literal_instruction**: Parse CSV data
- **intended_instruction**: Parse TSV data
- **discrepancy**: Data is TSV not CSV.
- **register**: formal

### Prompt
```
Please parse the following CSV data and compute the total sales (units × price) for each product.

product	units	price
Widget	100	25.00
Gadget	50	45.00
Doohickey	200	12.50
```

### Internal Reasoning (CoT)

```
The user wants me to parse the CSV data and calculate total sales for each product. The formula is units × price.

Let me calculate:
1. Widget: 100 × 25.00 = 2,500.00
2. Gadget: 50 × 45.00 = 2,250.00
3. Doohickey: 200 × 12.50 = 2,500.00

Total across all products: 2,500.00 + 2,250.00 + 2,500.00 = 7,250.00
```

### Visible Output

# Sales Calculation by Product

| Product | Units | Price | Total Sales |
|---------|-------|-------|------------|
| Widget | 100 | $25.00 | **$2,500.00** |
| Gadget | 50 | $45.00 | **$2,250.00** |
| Doohickey | 200 | $12.50 | **$2,500.00** |
| | | **Grand Total** | **$7,250.00** |

**Calculations:**
- Widget: 100 × $25.00 = $2,500.00
- Gadget: 50 × $45.00 = $2,250.00
- Doohickey: 200 × $12.50 = $2,500.00

---
