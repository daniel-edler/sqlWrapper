#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2016 - 2019, 2023 Daniel Edler
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
Read and write your data to a relational database like sqlite and (at least in parts) postgre.

This module provides a numpy adapter and a few convenient functions. You can freely choose to either
save your numpy objects as binary BLOB *into* the database (for smaler files) or as files on your
machine (for larger file). But this module can only deal with internal OR external stored numpy
data.

Minimal working example
========================
import sqlWrapper as sql

db = sql.iodb("sqlite", "intern")
db.start(filename="./myFirstDatabase.sqlite")

db.createTable("data", {"col1": 0.0, "description": "dummyText", "binary": np.array([])})
    # Or insert it directly with more low level methods without the use of a dictionary. Not the
    # shape or dtype but the fact that it's a numpy object is important

rownumber = db.insertInto("data", {"col1": 3.141, "description": "pi", "binary": np.array([3, 1, 4, 1])})
    # Or insert it directly with more low level methods without the use of a dictionary

db.execute("SELECT * FROM data WHERE col1 > 3 ORDER BY col1")

for dbRow in db.fetchall():
    print(dbRow)

(Driver specific) Limitations
=============================

* A mixed mode in postgre is not implemented. One has to define a complete new postgres datatype and
  I don't know if this is possible
"""


# TODO @createTable: not only determine the data type but also the value could also be seen as a default value

# stdlib modules
import sqlite3
import io
import types
# import socket # You might comment these lines out if you do not use the external storage of your
# import os     # database adapter.  They are needed to determine a unique filename over the whole cluster.
import time
import inspect # If callable is given. Save its definition as string instead
import gzip

# third party modules
import numpy as np
try:
    import psycopg2 # postgres
except ImportError:
    print("Cannot find postgres module. Consequently you should not try to use it")
#except ModuleNotFoundError: # python3 >= 3.6
    #print("Cannot find postgres module. Consequently you should not try to use it")

TIME_RECONNECT = 10 # How many seconds until a database reconnect after `cur.execute` failed and
                    # throwed an `sql.OperationalError`
                    # @sqlite: May occur when multiple processes access the database simultanously
                    #          (and it's on a network filesystem, namely NFS) => large number (30?)
                    # @postgres: Connection to server closed in the meantime. => small number should
                    #            be sufficient

class sqliteConnection(sqlite3.Connection):
    """ Inherited from sqlite3.Connection. Adds the delete function. Usefull for externally
    stored files """

    def delete(self, file_cols, db_from, db_filter):
        """ Similar to ??DELETE FROM ... WHERE?? SQL clause.

        file_cols (string) : columns which are of type TEXT and contain path information. These
                             files will be deleted before the rows in the database are deleted
        db_from (string) : DELETE FROM
        db_filter (string) : WHERE condition (without the word WHERE)
        """
        # Get a list of all files
        cur = self.cursor()
        cur.execute("SELECT {2} from {0} WHERE {1}".format(db_from, db_filter, file_cols))

        # unpacking (in *-notation is not possible with list comprehension)
        fnames = [ i for j in cur.fetchall() for i in j ]
        print(fnames)

        # delete the files
        for fname in fnames:
            if os.path.exists(fname + ".npy"):
                os.remove(fname + ".npy")
            else:
                print("delete: No such file: '{}'".format(fname + ".npy"))
                print("delete: The db record will be deleted anyway")

        # delete the database entries
        print("delete: Remember to commit() the changes to the database")
        return cur.execute("DELETE FROM {0} WHERE {1}".format(db_from, db_filter))

class ndarray_ext:
    """ Dummy class. Basically just a container to distinguish between internal and external arrays"""
    def __init__(self, arr):
        self.data = arr

# This class was greatly inspired by http://stackoverflow.com/questions/18621513 and
# my post here http://stackoverflow.com/questions/10529351/43983106#43983106
class iodb:
    """
    Class with a sql instance which can either be a sqlite3 or a postgreSQL object.
    Enriched by:

        * numpy adapters (internal BLOB or path-to-file)
            * internal BLOB or path-fo-file
        * convenient functions `start`, `createTable` and `insertInto` which take care of the
          different db syntaxes. I.e. just give `insertInto` a dictionary
    """

    def __init__(self, driver, storage, storagePath="./", loadZipped=False, readOnly=False):
        """
        driver : (string) Which database system. "sqlite" and "postgre" is supported
        storage : (string) Register the needed adapter for storing numpy array. Possible values:
                  "intern": np.ndarray are stored in the database as a BLOB of type ARRAY
                  "extern": sqlWrapper.ndarray_ext are in files on your filesystem. The db entry
                            is just a string path but registered as type ARRAY_EXT
                  "mixed": Both objects can be stored
        storagePath : (optional, string) Storage base folder. Must include an ending '/'
        readOnly : (optional, bool) Do not create a folder with the current date
        loadZipped : (optional, bool) In case of external data. Load the zipped (additional ".gz"
                     extention) file instead. Saves bandwidth and but consumes more time
        """
        # Driver settings
        driverlist = ["sqlite", "postgre"]
        if driver in driverlist:
            self.driver = driver
            if driver == "sqlite":
                self.sql = sqlite3
                self._timestamp = "timestamp datetime default current_timestamp"
                self._rowid = "rowid"
                self._arrayType = "ARRAY"
                self._arrayType_ext = "ARRAY_EXT"

                # Register the numpy integer datatypes.  np.floatXX are fine
                self.sql.register_adapter(np.int64, int)
                self.sql.register_adapter(np.int32, int)
                self.sql.register_adapter(np.int16, int)
                #self.sql.register_adapter(np.bool_, int)

                # How to save a function or method
                self.sql.register_adapter(types.FunctionType, self._setAdapter_lambda)
                self.sql.register_adapter(types.MethodType, self._setAdapter_lambda)
            elif driver == "postgre":
                ## How to use adapters: http://initd.org/psycopg/docs/advanced.html#adapting-new-types

                self.sql = psycopg2
                self._timestamp = "timestamp timestamp default current_timestamp"
                self._rowid = "oid"
                if storage == "intern":
                    self._arrayType = "BYTEA"
                else:
                    self._arrayType = "TEXT"
                self._arrayType_ext = "TEXT"

                # Register the numpy integer datatypes.  np.floatXX are fine
                self.sql.extensions.register_adapter(np.int64, self.sql.extensions.Int)
                self.sql.extensions.register_adapter(np.int32, self.sql.extensions.Int)
                self.sql.extensions.register_adapter(np.int16, self.sql.extensions.Int)
                #self.sql.extensions.register_adapter(np.bool_, self.sql.extensions.Boolean)

                # How to save a function or method
                self.sql.extensions.register_adapter(types.FunctionType, self._setAdapter_lambda)
                self.sql.extensions.register_adapter(types.MethodType, self._setAdapter_lambda)
        else:
            print("Unknown database driver '{0}'. Possible driver are {1}".format(
                driver, driverlist))

        self.loadZipped = loadZipped
        self.storage = storage
        self._spath = storagePath
        if storage == "extern" or storage == "mixed":
            self._counter = 1
            self._folder = time.strftime("%Y-%m-%d") + "/"
            fullpath = self._spath + self._folder

            # create folder
            if not os.path.exists(fullpath) and not readOnly:
                os.makedirs(fullpath)

    def _setAdapter_lambda(self, obj):
        """ Handles how to store a callable object """
        if self.driver == "sqlite":
            return inspect.getsource(obj).strip()
        else:
            sourcecode = inspect.getsource(obj).strip()
            return self.sql.adapt(sourcecode)

    def _setAdapters(self, storage):
        if storage == "extern":
            # Register the external adapter for storing a numpy object as a file in the filesystem
            self._setAdapterIn_extern(self._arrayType)
            self._setAdapterOut_extern(np.ndarray)
            self._setAdapterIn_extern(self._arrayType_ext) # Just to be sure
            #self._setAdapterOut_extern(ndarray_ext)        # Just to be sure
        elif storage == "mixed":
            self._setAdapterIn_extern(self._arrayType_ext)
            self._setAdapterOut_extern(ndarray_ext)
            self._setAdapterIn_intern(self._arrayType)
            self._setAdapterOut_intern(np.ndarray)
        elif storage == "intern" or storage == "mixed":
            # Register the internal adapter for storing a numpy object directly as a BLOB
            self._setAdapterIn_intern(self._arrayType)
            self._setAdapterOut_intern(np.ndarray)

    def _setAdapterOut_intern(self, blob=np.ndarray):
        """ Convert from numpy to BLOB """
        if self.driver == "sqlite":
            self.sql.register_adapter(blob, self._adapt_array)
        elif self.driver == "postgre":
            self.sql.extensions.register_adapter(blob, self._adapt_array)

    def _setAdapterOut_extern(self, blob=np.ndarray):
        """ Convert from numpy to BLOB """
        if self.driver == "sqlite":
            self.sql.register_adapter(blob, self._adapt_array_ext)
        elif self.driver == "postgre":
            self.sql.extensions.register_adapter(blob, self._adapt_array_ext)

    def _setAdapterIn_intern(self, typeName="ARRAY"):
        """ Convert from BLOB to numpy """
        if self.driver == "sqlite":
            self.sql.register_converter(typeName, self._typecast_array)
        elif self.driver == "postgre":
            args = self.sql.BINARY.values, typeName, self._typecast_array

            t_array = self.sql.extensions.new_type(*args)
            self.sql.extensions.register_type(t_array)

    def _setAdapterIn_extern(self, typeName="ARRAY_EXT"):
        """ Convert from BLOB to numpy """
        if self.driver == "sqlite":
            self.sql.register_converter(typeName, self._typecast_array_ext)
        elif self.driver == "postgre":
            args = self.sql.STRING.values, typeName, self._typecast_array_ext

            # self.sql.extensions.register_type
            t_array = self.sql.extensions.new_type(*args)
            self.sql.extensions.register_type(t_array)

    def _adapt_array(self, arr):
        """ Converts from numpy to postgres or sqlite BLOB"""
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return self.sql.Binary(out.read())

    def _typecast_array(self, value, cur=None):
        """ Converts from postgres or sqlite BLOB to numpy. In case of postgres a cur is given """
        if value is None and cur is not None:
            return None

        if cur is not None:
            value = self.sql.BINARY(value, cur)

        bdata = io.BytesIO(value)
        bdata.seek(0)  # rewind to first position in the buffer
        return np.load(bdata)

    def _adapt_array_ext(self, arr):
        """ Converts from numpy object to numpy-binary-file """
        if type(arr) == ndarray_ext:
            arr = arr.data

        # Get the filename from the class variable `binaryNames` or create one
        if "binaryNames" in dir(self):
            if len(self.binaryNames) > 0:
                fname, self.binaryNames = self.binaryNames[0], self.binaryNames[1:]
            else:
                fname = self._getUniqueName()
        else:
            fname = self._getUniqueName()

        # Save the file to the filesystem
        np.save(self._spath + self._folder + fname, arr)

        if self.driver == "sqlite":
            return self._folder + fname
        else:
            return self.sql.extensions.adapt(self._folder + fname)
            #return self.sql.extensions.AsIs("'"+self._folder + fname+"'")

    def _typecast_array_ext(self, value, cur=None):
        """ Converts from numpy-binary-file to numpy object. In case of postgres a cur is given """
        if value is None and cur is not None:
            return None

        if cur is not None:  # so it is a postgres database
            value = self.sql.STRING(value, cur)
        else:
            value = value.decode("utf-8")

        fname = self._spath + value + ".npy"
        if self.loadZipped and os.path.isfile(fname + ".gz"):
            fileObj = gzip.open(fname + ".gz", "rb")
            out = np.load(fileObj)
        elif os.path.isfile(fname):
            out = np.load(fname)
        else:
            raise FileNotFoundError("No such file or directory '{0}'".format(fname))

        return out

    def _getUniqueName(self):
        """ Return a unique name consisting of date, time, process id and a counter """
        hostname = socket.gethostname()
        pid = os.getpid()
        curDate = time.strftime("%Y-%m-%d")
        curTime = time.strftime("%H%M")

        fname = "{0}T{1}_{2}_{3}_{4}".format(
            curDate, curTime, hostname, pid, self._counter)
        self._counter += 1

        return fname

    def start(self, **kwargs):
        """ Establish a connection (con) and create a cursor (cur).  Install numpy adapter

        Parameter
        ---------
        filename : (string) @sqlite Filename and optional the location of the database
        database : (string) @postgre Database name
        user : (string) @postgre Username
        password : (string) @postgre Password corresponding to the username
        host : (string) @postgre Either the unix socket location (local) or computer address
        port : (int) @postgre The port number
        """
        if self.driver == "sqlite":
            self._setAdapters(self.storage)
            self._fname = kwargs["filename"]
            # the detect_types argument is important
            self.con = self.sql.connect(self._fname, detect_types=self.sql.PARSE_DECLTYPES)
        elif self.driver == "postgre":
            self.con = self.sql.connect(**kwargs)

            self._setAdapters(self.storage)

        self._conKwargs = kwargs
        self.cur = self.con.cursor()

    def stop(self):
        """ Close connection and cursor """
        self.cur.close()
        self.con.close()

    def select(self, dbcols, dbfrom, dbfilter="", dbother="", verbose=False, nosize=False, *args, **kwargs):
        """ similar to execute() but prepends a count query and returns this number. Useful when
        you need to store the database returned values in a (allocated) array

        Parameter
        ---------
        dbcols : (str) names of the columns in the table you want to select
        dbfrom : (str) table name where the columns are located
        dbfilter : (str, optional) SQL where condition without the `where` keyword
        dbother : (str, optional) other SQL stuff like `order by counter`
        nosize : (bool) do not count the number of returned rows

        Return
        ------
        size : number of rows ("records" in sql term)
        data : sql.iterator of the returned data
        """
        if dbfilter.strip() != "": dbfilter = " WHERE " + dbfilter
        query = "select {} from {} {} {}".format(dbcols, dbfrom, dbfilter, dbother)
        if verbose: print(query)

        if nosize:
            return None, self.execute(query)
        else:
            self.execute("SELECT COUNT(*) FROM ({})".format(query))
            size = self.cur.fetchone()[0]

            return size, self.execute(query)

    def execute(self, *args, **kwargs):
        """ Almost same as cur.execute """
        try:
            return self.cur.execute(*args, **kwargs)
        except self.sql.OperationalError as err:
            print(err)
            print(*args)
            raise err
        # except self.sql.OperationalError as err: # psycopg2 error when server was closed. Try again
        #     # TODO is also thrown in many other cases e.g. when a column does not exists
        #     print("WARNING:", err)
        #     print("NOTE: Try to reconnect after {}s".format(TIME_RECONNECT))
        #     time.sleep(TIME_RECONNECT)
        #     self.con = self.sql.connect(**self._conKwargs)
        #     self.cur = self.con.cursor()
        #     return self.cur.execute(*args, **kwargs)

    def fetchone(self, *args, **kwargs):
        """ same as cur.fetchone """
        return self.cur.fetchone(*args, **kwargs)

    def fetchall(self, *args, **kwargs):
        """ same as cur.fetchall """
        return self.cur.fetchall(*args, **kwargs)

    def fetchallT(self):
        """ Return all values like in fetchall but transposed. Each column is an np.ndarray """
        fetched = self.cur.fetchall()
        return [ np.array(col) for col in zip(*fetched) ]

    def lock(self, suffix=".lock", repeat=0, timeoutCounter=1000):
        """ Create a lock file

        Parameter
        =========
        suffix : Suffix of the lockfile
        repeat : Try again after `repeat` ms. Zero means single run

        Return
        ======
        success : `None` if postgre, `True` if successfull, `False` if failed (only if repeat == 0)
        """
        if self.driver == "sqlite":
            lockfilename = self._fname + suffix

            counter = 0
            while(True):
                if os.path.exists(lockfilename):
                    success = False
                else:
                    os.mknod(lockfilename)
                    success = lockfilename

                ## prepare for next loop
                if repeat > 0:
                    if counter > timeoutCounter:
                        print("WARNING:iodb:lock: Cannot create new lock file. Timeout reached")
                        break
                    else:
                        counter += 1
                        time.sleep(repeat/1000)
                else:
                    break

            return success

    def release(self, lockfilename=None, suffix=".lock"):
        """ Deletes the lock file if present

        Parameter
        =========
        suffix : Suffix of the lockfile. Useless when `lockfilename` is specified
        lockfilename : Complete filename of the lock file. Disables the `suffix` kwarg

        Return
        ======
        success : `None` if postgre, `True`(/`False`) on success(/failure)
        """
        if self.driver == "sqlite":
            if lockfilename is None:
                lockfilename = self._fname + suffix

            if os.path.exists(lockfilename):
                os.remove(lockfilename)
                return True
            else: return False

    def save(self, unlock=False):
        """ Commit all changes """
        self.con.commit()
        if unlock:
            self.release()

    def detectType(self, value, dbdriver):
        """ Depending on the input value (try to) find the appropriate database type.
        This depends on the database driver (sqlite or postgre) """
        # bool(value) is instance of int so first ask if bool(value) is instance of bool
        if isinstance(value, bool):
            if dbdriver == "sqlite":
                return "INTEGER"
            elif dbdriver == "postgre":
                return "BOOL"

        elif isinstance(value, (int, np.int32, np.int16)):
            return "INTEGER"
        elif isinstance(value, np.int64):
            if dbdriver == "sqlite":
                return "INTEGER"
            elif dbdriver == "postgre":
                return "BIGINT"
        elif isinstance(value, np.int8):
            if dbdriver == "sqlite":
                return "INTEGER"
            elif dbdriver == "postgre":
                return "SMALLINT"

        elif isinstance(value, str):
            if value == "ndarray":
                print("iodb::detectType: (internal) numpy array detected", value)
                return self._arrayType
            elif value == "ndarray_ext":
                print("iodb::detectType: (external) numpy array detected", value)
                return self._arrayType_ext
            return "TEXT"

        elif isinstance(value, (float, np.float64)):
            if dbdriver == "sqlite":
                return "REAL" # 8-byte float
            elif dbdriver == "postgre":
                return "DOUBLE PRECISION" # also possible: REAL (= float4)
        elif isinstance(value, (np.float16, np.float32)):
            return "REAL"

        elif isinstance(value, np.ndarray):
            if type(value) == ndarray_ext:
                return self._arrayType_ext
            return self._arrayType

        elif callable(value):
            return "TEXT" # Save the function name, source code or similar

        elif isinstance(value, list) or isinstance(value, tuple):
            if len(value) == 1:
                war = "list or tuple with exactly one element detected.\n"
                war += " "*27 + "I'll save just the single element: {}".format(value)
                print("WARNING: iodb::detectType:", war)
                return self.detectType(value[0], dbdriver)
            else:
                err = "iodb::detectType: Unrecognized type '{}'\nWith value: {}".format(
                    type(value), value)
                raise TypeError(err)

        else:
            err = "iodb::detectType: Unrecognized type '{}'\nWith value: {}".format(
                    type(value), value)
            raise TypeError(err)

    def appendType(self, indict, useTypes, sortBefore=True):
        """ Create a SQL string in the form "c1 type1, c2 type2". If the database is sqlite
        (driver == "sqlite") and useTypes is False then the type is omitted """
        keys = indict.keys()
        if sortBefore:
            keys = np.sort(list(keys))

        if useTypes is False and self.driver == "sqlite":
            return ", ".join(keys)
        else:
            parameterNames = []
            for key in keys:
                valtype = self.detectType(indict[key], self.driver)
                parameterNames.append(f"'{key}' {valtype}")
            return ", ".join(parameterNames)

    def getTableSchema(self, tblname):
        try:
            self.execute(f"PRAGMA table_info({tblname})")
        except sqlite3.OperationalError as err:
            print(f"Check the View definition if this is one ({tblname}). Is there a faulty column?")
            raise sqlite3.OperationalError(err)
            names, types = [], []
        else:
            cid, names, types, notnull, dflt_value, pk = list(zip(*self.fetchall()))

        return dict(zip(names, types))

    def getTableNames(self, types=["table"], verbose=False):
        allTypes = ", ".join([ f"'{i}'" for i in types ])
        query = f"SELECT tbl_name FROM sqlite_master WHERE type in ({allTypes})"
        if verbose: print(query)

        self.execute(query)
        ret = self.fetchall()

        if ret is None:
            return []

        if len(ret) == 0:
            return []

        tblnames, = list(zip(*ret))
        return tblnames

    def getAllTableSchema(self, types=["table"], verbose=False):
        # Get all tables
        tblnames = self.getTableNames(types)

        if len(tblnames) == 0:
            return {}

        out = {}
        for tblname in tblnames:
            out[tblname] = self.getTableSchema(tblname)

        return out

    def createTable(self, tablename, indict,
                    useTypes=True, save=True, before="", after="", verbose=False,
                    sortBefore=True):
        """
        Create a database table called `tablename` with the column names given in the
        dictionary `indict`. Depending on the corresponding value an appropriate database type is
        chosen as long as useTypes is True.
        TODO Add default values

        Parameter
        ---------
        indict : Sample input dictionary in the style {'key-as-used-in-sql-column-name': value}
                 where `value` should have the same datatype as the expected data
        useTypes : Determine the type of the dict-value and try to map it to a db equivalent. In
                   the case of SQLite it forces to use types. Otherwise this option has no meaning
        save : Write the changes (created table) to the database?
        sortBefore : Since dictionaries don't have ordered keys, should the keys be sorted before?
        before : ready to use SQL string before the given columns with closing ', '
        after : ready to use SQL string after the given columns with starting ', '
        """
        parameterNames = self.appendType(indict, useTypes, sortBefore)

        ## @sqlite: an implicit column with the name rowid/_rowid_/oid is always created and usually
        ## hidden when you query for SELECT * from tablename. It is a per table unique, monotonic
        ## increasing integer number. Perfect for joining on other tables (primary key).
        ## @postgres: nothing really equivalent to rowid/_rowid/oid exists. You can create a
        ## table with WITH OID but this feature will be removed in postgres 12 and still is not
        ## exactly what you want. Manually create a sequence instead (but which is not hidden by
        ## SELECT * from tablename)
        if self.driver == "sqlite":
            query = "CREATE TABLE IF NOT EXISTS {} ({})".format(
                tablename, before + parameterNames + after)
        elif self.driver == "postgre":
            query = "CREATE TABLE IF NOT EXISTS {} ({} SERIAL PRIMARY KEY, {})".format(
                tablename, self._rowid, before + parameterNames + after)

        if verbose: print(query)

        self.execute(query)

        if save:
            self.save()

    def insertInto(self, tablename, indict, save=True, verbose=False, binaryDict={}, returnLastRow=True):
        """
        Insert into `tablename` the values in the dictionary `indict`

        Parameter
        ---------
        tablename : (string) name of the table
        indict : (dict) dictionary in the form {table-column-name: value-or-object}
        save : (optional, bool) if true then commit changes to the database in the ending
        verbose : (optional, bool) print the executed queries. (Good for debugging)
        # binaryDict : (optional, dict) when the storage is external it specifies the filename.
        #              It is in the form {table-column-name: filename}.  In the case of an
        #              empty dictionary a generic unique name is used
        returnLastRow : (optional, bool) return the rowid of the newly added row/record
        """
        keys = indict.keys()
        values = [ indict[key] for key in keys ]
        keys_sqlsafe = [ f"'{i}'" for i in keys ]

        # Create a list of binaryDict content (so the file names of the binary files) where the
        # elements are in exactly in the same order of occurence as in keys
        # self.binaryNames = []
        # if len(binaryDict) != 0:
        #     if not keys.isdisjoint(binaryDict):
        #         for key in keys:
        #             if key in binaryDict:
        #                 self.binaryNames.append(binaryDict[key])

        if self.driver == "sqlite":
            questionmarks = ", ".join(["?"] * len(keys))
            query = "insert into {0} ({1}) values ({2})".format(
                                tablename, ", ".join(keys_sqlsafe), questionmarks)

            if verbose: print(query)
            self.execute(query, values)

            if returnLastRow:
                lastrowid = self.cur.lastrowid
        else:
            nomarks = ", ".join(["%s"] * len(keys))
            query = "insert into {0} ({1}) values ({2}) returning {3}".format(
                                tablename, ", ".join(keys_sqlsafe), nomarks, self._rowid)

            if verbose: print(query)
            self.execute(query, values)

            if returnLastRow:
                lastrowid = self.cur.fetchone()[0]

        if save:
            self.save()

        if returnLastRow:
            return lastrowid
