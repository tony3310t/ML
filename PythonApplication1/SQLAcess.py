import pymysql

#宣告共用Constant-------------
_host="fintech.myds.me"
_port=3307
_user="wordpress_user"
_password="folk1230"
_db="wordpress"
#-------------------------------


#取回Table 資料 ----------------------
def GetData(SQL):
    try:
        db = pymysql.connect(host=_host,port=_port,user=_user,password=_password,db=_db,charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor )
        cursor = db.cursor()
        cursor.execute(SQL)
        results = cursor.fetchall()                     
    except:
        print("error")

    finally:
        db.close()
    return results
#---------------------------------------

#執行 Insert Delete 使用Function
def ExecutSQL(SQL):
    try:
        result = True
        db = pymysql.connect(host=_host,port=_port,user=_user,password=_password,db=_db,charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor )
        cursor = db.cursor()
        cursor.execute(SQL)
        results = cursor.fetchall()                     
    except:
        print("error")
        result = False
    finally:
        db.close()
    return result