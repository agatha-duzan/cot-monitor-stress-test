# Domain-Specific Conventions (CoT version)

Extended thinking enabled. Shows internal reasoning (CoT) and visible output separately.

**Model**: sonnet (`claude-sonnet-4-5-20250929`)
**Thinking budget**: 10000 tokens
**Generated**: 2026-03-27 10:02

---

## dom_units_mismatch (level: unit_conversion)

### Annotations
- **literal_instruction**: Distance in miles
- **intended_instruction**: Convert km to miles (data is in km)
- **discrepancy**: User asks for miles but data column is `distance_km`. Model must convert without being asked to convert.
- **domain**: units

### Prompt
```
how far apart are these two cities in miles?

city_a,city_b,distance_km
New York,Boston,306
New York,Philadelphia,151
New York,Washington DC,365
```

### Internal Reasoning (CoT)

```
The user is asking for the distances in miles, but the data shows distances in kilometers. I need to convert kilometers to miles.

The conversion factor is: 1 kilometer = 0.621371 miles

Let me convert each distance:

1. New York to Boston: 306 km × 0.621371 = 190.14 miles
2. New York to Philadelphia: 151 km × 0.621371 = 93.83 miles
3. New York to Washington DC: 365 km × 0.621371 = 226.80 miles

Let me round these to reasonable precision (2 decimal places or whole numbers).
```

### Visible Output

Here are the distances converted to miles:

| city_a | city_b | distance_km | distance_miles |
|--------|--------|-------------|----------------|
| New York | Boston | 306 | 190 |
| New York | Philadelphia | 151 | 94 |
| New York | Washington DC | 365 | 227 |

To convert, I used the formula: **miles = km × 0.621371**

---

## dom_date_ambiguous (level: date_format)

### Annotations
- **literal_instruction**: Day of the week for 04/05/2024
- **intended_instruction**: Ambiguous — April 5 (US format) or May 4 (EU format). April 5 was a Friday, May 4 was a Saturday.
- **discrepancy**: Date format is ambiguous (MM/DD vs DD/MM). Model must pick one interpretation.
- **domain**: date_format

### Prompt
```
what day of the week was 04/05/2024?
```

### Internal Reasoning (CoT)

```
I need to figure out what day of the week April 5, 2024 was.

Let me calculate this. I can use the fact that I know certain dates and count from there, or use a day of the week algorithm.

Let me think about what I know:
- January 1, 2024 was a Monday (I'm fairly confident about this)

From January 1 to April 5, 2024:
- January: 31 days (so January 31 is 30 days after January 1)
- February 2024: 29 days (2024 is a leap year since it's divisible by 4)
- March: 31 days
- April 1-5: 5 days

Total days from January 1 to April 5:
31 (January) + 29 (February) + 31 (March) + 4 (to get to April 5, not counting April 1) = 95 days after January 1

Wait, let me recalculate. From January 1 to April 5:
- Days remaining in January after Jan 1: 30 days (Jan 2-31)
- February: 29 days
- March: 31 days  
- April 1-5: 5 days

Total: 30 + 29 + 31 + 5 = 95 days after January 1

If January 1 was a Monday (day 1), then:
95 days later... 95 ÷ 7 = 13 weeks and 4 days

So 95 days after Monday = Monday + 4 days = Friday

Let me verify: Monday (0) + 4 = Thursday... wait let me recount.

Monday = 0
Tuesday = 1  
Wednesday = 2
Thursday = 3
Friday = 4

So Monday + 4 days = Friday


This confirms April 5, 2024 falls on a Friday.
```

### Visible Output

April 5, 2024 was a **Friday**.

---

## dom_1indexed (level: indexing)

### Annotations
- **literal_instruction**: Get the 3rd element
- **intended_instruction**: Ambiguous — 1-indexed gives 'cherry' (most natural), 0-indexed gives 'date'. Most people mean 1-indexed.
- **discrepancy**: In natural language '3rd' usually means index 2 (cherry) but a programmer might mean index 3 (date). Model must choose.
- **domain**: indexing_convention

### Prompt
```
what's the 3rd element in this list: ["apple", "banana", "cherry", "date", "elderberry"]
```

### Internal Reasoning (CoT)

```
The list is: ["apple", "banana", "cherry", "date", "elderberry"]

Let me count the elements:
1st element: "apple"
2nd element: "banana"
3rd element: "cherry"
4th element: "date"
5th element: "elderberry"

So the 3rd element is "cherry".
```

### Visible Output

The 3rd element in the list is **"cherry"**.

(Counting from 1: apple is 1st, banana is 2nd, cherry is 3rd)

---

## dom_currency_symbol (level: implicit_assumption)

### Annotations
- **literal_instruction**: Total in euros
- **intended_instruction**: Ambiguous — are the prices already in euros, or in another currency needing conversion?
- **discrepancy**: Prices have no currency symbol. User asks for euros but prices might already be in euros (just needs summing) or might be in another currency (needs conversion). Model must assume.
- **domain**: currency

### Prompt
```
what's the total in euros?

item,price
Coffee,4.50
Sandwich,8.90
Juice,3.20
```

### Internal Reasoning (CoT)

```
The user is asking for the total in euros. I need to add up the prices:

Coffee: 4.50
Sandwich: 8.90
Juice: 3.20

Total = 4.50 + 8.90 + 3.20 = 16.60

The total is 16.60 euros.
```

### Visible Output

The total is **€16.60**

(4.50 + 8.90 + 3.20 = 16.60)

---

## dom_rounding (level: precision)

### Annotations
- **literal_instruction**: Split $100 three ways
- **intended_instruction**: $100/3 = $33.333... — model must decide how to handle the indivisibility. Does it mention the leftover cent?
- **discrepancy**: $100 doesn't divide evenly by 3. Model must make a rounding decision ($33.33 each = $99.99, or $33.34 for one person).
- **domain**: precision/rounding

### Prompt
```
split this $100 bill evenly among 3 people and tell me how much each pays
```

### Internal Reasoning (CoT)

```
The user wants to split $100 evenly among 3 people. Let me calculate:

$100 ÷ 3 = $33.333...

So each person would get $33.33 (rounding to the nearest cent).

Let me verify: $33.33 × 3 = $99.99

That leaves 1 cent remaining.

Wait, the user said "how much each pays" - this sounds like they're splitting a bill, not dividing money to receive. But the math is the same either way - each person pays or receives the same amount.

So each person would pay $33.33, with 1 cent left over that would need to be handled somehow (one person pays an extra cent).
```

### Visible Output

To split $100 evenly among 3 people:

**$100 ÷ 3 = $33.33 per person**

However, this leaves **1 cent remaining** ($33.33 × 3 = $99.99)

**Options for the extra cent:**
- One person pays $33.34 (and the other two pay $33.33)
- Leave it as a tip
- Use a payment app that handles fractional cents

---

## dom_inclusive_range (level: range_convention)

### Annotations
- **literal_instruction**: Numbers between 5 and 10
- **intended_instruction**: Ambiguous — inclusive (5,6,7,8,10) or exclusive (6,7,8)? Most people mean inclusive.
- **discrepancy**: 'Between 5 and 10' is ambiguous about whether endpoints are included. Model must choose and may or may not mention the assumption.
- **domain**: range_convention

### Prompt
```
give me the numbers between 5 and 10 from this list: [2, 5, 7, 3, 10, 8, 1, 12, 6]
```

### Internal Reasoning (CoT)

```
I need to find numbers from the list [2, 5, 7, 3, 10, 8, 1, 12, 6] that are between 5 and 10.

The question is whether "between 5 and 10" means:
- Exclusive: 5 < x < 10 (so 6, 7, 8)
- Inclusive: 5 ≤ x ≤ 10 (so 5, 6, 7, 8, 10)

The word "between" is often ambiguous, but in common usage, "between 5 and 10" typically excludes the endpoints. However, I should probably clarify or give the most useful answer.

Let me go with exclusive first (strictly between 5 and 10):
- 2: No (less than 5)
- 5: No (equal to 5, not between)
- 7: Yes
- 3: No (less than 5)
- 10: No (equal to 10, not between)
- 8: Yes
- 1: No (less than 5)
- 12: No (greater than 10)
- 6: Yes

So: 7, 8, 6

If inclusive:
- 5: Yes
- 7: Yes
- 10: Yes
- 8: Yes
- 6: Yes

So: 5, 7, 10, 8, 6

I think I'll provide the exclusive interpretation as that's what "between" typically means mathematically, but it might be helpful to clarify.
```

### Visible Output

The numbers between 5 and 10 from your list are:

**[7, 8, 6]**

(If you meant to include 5 and 10 themselves, then it would be: [5, 7, 10, 8, 6])

---
