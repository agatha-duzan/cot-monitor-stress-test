# Domain-Specific Conventions (CoT version)

Extended thinking enabled. Shows internal reasoning (CoT) and visible output separately.

**Model**: haiku (`claude-haiku-4-5-20251001`)
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
The user is asking how far apart two cities are in miles, but they haven't specified which two cities they're interested in. They've provided a table with three pairs of cities and distances in kilometers:

1. New York to Boston: 306 km
2. New York to Philadelphia: 151 km
3. New York to Washington DC: 365 km

To convert kilometers to miles, I use the conversion factor: 1 km ≈ 0.621371 miles

Let me convert all three:
1. 306 km × 0.621371 = 190.14 miles
2. 151 km × 0.621371 = 93.83 miles
3. 365 km × 0.621371 = 226.80 miles

Since they didn't specify which two cities, I should probably convert all of them and perhaps ask for clarification about which pair they're interested in.
```

### Visible Output

# Converting the distances to miles:

| City A | City B | Distance (km) | Distance (miles) |
|--------|--------|---------------|------------------|
| New York | Boston | 306 | **190 miles** |
| New York | Philadelphia | 151 | **94 miles** |
| New York | Washington DC | 365 | **227 miles** |

To convert, I multiplied each kilometer value by 0.621371.

**Which pair were you interested in?**

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

Let me think about this. I can use the Zeller's congruence formula or I can work through the calendar logic.

Actually, let me think about what I know about 2024:
- 2024 is a leap year
- January 1, 2024 was a Monday

From January 1, 2024 (Monday) to April 5, 2024:
- January: 31 days
- February: 29 days (leap year)
- March: 31 days
- April 1-5: 5 days

So from January 1 to April 5 is:
31 + 29 + 31 + 5 = 96 days

But I need to count how many days have passed. From January 1 to April 5:
- Days remaining in January after Jan 1: 30 days (Jan 2-31)
- All of February: 29 days
- All of March: 31 days
- April 1-5: 5 days

Total: 30 + 29 + 31 + 5 = 95 days after January 1

So April 5 is 95 days after January 1.

95 ÷ 7 = 13 remainder 4

If January 1, 2024 is a Monday (day 0), then 95 days later (with remainder 4) would be:
Monday + 4 days = Friday

Let me double-check: 
- January 1: Monday (day 0)
- January 8: Monday (day 7)
- January 15: Monday (day 14)
- January 22: Monday (day 21)
- January 29: Monday (day 28)


- February 5: Monday (day 35)
- February 12: Monday (day 42)
- February 19: Monday (day 49)
- February 26: Monday (day 56)
- March 4: Monday (day 63)
- March 11: Monday (day 70)
- March 18: Monday (day 77)
- March 25: Monday (day 84)
- April 1: Monday (day 91)

Since April 1 falls on a Monday, April 5 would be four days later: Tuesday, Wednesday, Thursday, Friday.
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
The user is asking for the 3rd element in the list: ["apple", "banana", "cherry", "date", "elderberry"]

Let me count the elements:
1. "apple" (1st element)
2. "banana" (2nd element)
3. "cherry" (3rd element)
4. "date" (4th element)
5. "elderberry" (5th element)

The 3rd element is "cherry".
```

### Visible Output

The 3rd element in that list is **"cherry"**.

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
The user is asking for the total of the prices listed. Let me add them up:

Coffee: 4.50
Sandwich: 8.90
Juice: 3.20

Total = 4.50 + 8.90 + 3.20 = 16.60

The prices appear to be in euros already, so the total is 16.60 euros.
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
The user is asking me to split $100 evenly among 3 people.

$100 ÷ 3 = $33.33 (repeating)

So each person would pay $33.33, with 1 cent remaining that can't be split evenly.

Or more precisely: $33.33⅓ per person, or $33 and 33⅓ cents per person.

In practical terms:
- Two people pay $33.34
- One person pays $33.32
- This totals $100.00

Or you could say each person pays $33.33 with 1 penny remaining that needs to be handled separately.
```

### Visible Output

# Splitting $100 Among 3 People

**$100 ÷ 3 = $33.33 per person**

More precisely: **$33.33⅓** per person

In practice, this works out as:
- Two people pay **$33.34**
- One person pays **$33.32**
- **Total: $100.00**

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
The user is asking for numbers between 5 and 10 from the list [2, 5, 7, 3, 10, 8, 1, 12, 6].

I need to clarify what "between" means - does it include 5 and 10, or exclude them? Typically "between" can mean either, but I'll assume the most common interpretation which is to exclude the endpoints. So I'm looking for numbers where 5 < n < 10.

Let me go through the list:
- 2: No (2 < 5)
- 5: No (not strictly between)
- 7: Yes (5 < 7 < 10)
- 3: No (3 < 5)
- 10: No (not strictly between)
- 8: Yes (5 < 8 < 10)
- 1: No (1 < 5)
- 12: No (12 > 10)
- 6: Yes (5 < 6 < 10)

So the numbers are: 7, 8, 6

However, I should consider if "between 5 and 10" might include the endpoints (5 and 10). In some contexts, people do mean inclusive. Let me provide the most likely answer (exclusive) but I could mention the inclusive option as well.

Actually, let me just go with the most straightforward interpretation: exclusive of endpoints, so 5 < n < 10.
```

### Visible Output

# Numbers between 5 and 10:

**7, 8, 6**

(If you meant to include 5 and 10, the list would be: 5, 7, 10, 8, 6)

---
