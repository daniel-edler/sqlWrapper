# sqlWrapper
Read and write data in the form of a numpy object with Python to a relational database

This module provides a numpy adapter and a few convenient functions. You can freely choose to either
save your numpy objects as binary BLOB *into* the database (for smaler files) or as files on your
machine (for larger file). But this module can only deal with internal OR external stored numpy
data.

Minimal working example
========================
```python
import sqlWrapper as sql

db = sql.iodb("sqlite", "intern")
db.start(filename="./myFirstDatabase.sqlite")

db.createTable("data", {"col1": 0.0, "description": "dummyText", "binary": np.array([])})
```

Not the shape or dtype but the fact that it's a numpy object is important.
Or create it directly with more low level methods without the use of a dictionary. The sqlite connection is `db.con` and the cursor `db.cur`, respectively.

```python
rownumber = db.insertInto("data", {"col1": 3.141, "description": "pi", "binary": np.array([3, 1, 4, 1])})
    # Or insert it directly with more low level methods without the use of a dictionary

db.execute("SELECT * FROM data WHERE col1 > 3 ORDER BY col1")

for dbRow in db.fetchall():
    print(dbRow)
```

(Driver specific) Limitations
=============================

* A mixed mode in postgre is not implemented. One has to define a complete new postgres datatype and
  I don't know if this is possible

See also
========

 * https://github.com/daniel-edler/blopy Extension for sqlite to see some numpy infos like size and data type. Or return a string representation of the some array elements
