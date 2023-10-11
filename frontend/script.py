import random
from datetime import date, timedelta
employees = [    
    "91021111",
    "91021151",
    "10921212",
    "48234012",
    "43204121",
     "1823012",
    "13220812" 
    ]

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        day = start_date + timedelta(n)
        if day.weekday() < 5:
            yield day

query = ""
counter = 1

start_date = date(2022, 1, 1)
end_date = date(2023, 1, 1)
for single_date in daterange(start_date, end_date):
    day = single_date.strftime("%Y-%m-%d")
    for emp in employees:
        line = f"({10000001+counter}, {emp}, DATE '{day}', {random.randint(3,8)}),\n"
        query += line
        counter+=1
print(query)
