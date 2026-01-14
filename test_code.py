from db.dbclass import DBClass


if __name__ == '__main__':
    db = DBClass()
    collection = db.BTCUSDTPCollection

    doc = collection.find_one()
    print(doc)