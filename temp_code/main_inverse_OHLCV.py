from datetime import datetime, timedelta
import json
from db.dbclass import DBClass
from pymongo import UpdateOne
from tqdm import tqdm

def datetime_to_dict(start_str, end_str):
    # [수정 1] 타임존 정보 제거 (.replace(tzinfo=None))
    # DB에서 나오는 데이터가 Naive 타입이므로, 딕셔너리 키도 Naive로 맞춰줍니다.
    start_datetime = datetime.fromisoformat(start_str).replace(tzinfo=None)
    end_datetime = datetime.fromisoformat(end_str).replace(tzinfo=None)

    # 2. datetime_list 만들기 (5분 간격)
    datetime_list = []
    current_time = start_datetime

    while current_time <= end_datetime:
        datetime_list.append(current_time)
        current_time += timedelta(minutes=5)

    # 3. inverse_datetime_list 만들기 (역정렬)
    inverse_datetime_list = datetime_list[::-1]

    # 4. datetime_dict 만들기
    datetime_dict = {}
    for f_datetime, i_datetime in zip(datetime_list, inverse_datetime_list):
        datetime_dict[f_datetime] = i_datetime
    
    return datetime_dict

if __name__ == "__main__":
    start_str = "2022-06-01T00:00:00.000+00:00"
    end_str = "2025-10-31T23:55:00.000+00:00"
    
    db = DBClass()
    collection = db.BTCUSDTP_5MinCollection
    target_collection = db.INVERSED_BTCUSDTP_5MinCollection
    
    # 딕셔너리 생성
    datetime_dict = datetime_to_dict(start_str, end_str)
    
    # DB 쿼리용 날짜도 타임존 제거하여 매칭
    start_dt = datetime.fromisoformat(start_str).replace(tzinfo=None)
    end_dt = datetime.fromisoformat(end_str).replace(tzinfo=None)

    # MongoDB 쿼리
    query = {
        "datetime": {
            "$gte": start_dt,
            "$lte": end_dt
        }
    }

    # [수정 2] Projection 유지 (_id는 0이므로 doc['_id'] 사용 불가)
    projection = {
        "_id": 0,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
        "sum_open_interest_value": 1,
        "datetime": 1
    }

    cursor = collection.find(query, projection)

    operations = []
    
    # tqdm 사용 시 total 개수를 알면 진행바가 더 정확해집니다 (선택사항)
    # total_count = collection.count_documents(query)
    # for doc in tqdm(cursor, total=total_count):
    
    for doc in tqdm(cursor):
        # 방어 코드: DB 데이터가 범위 밖이거나 dict에 없는 경우 건너뜀
        current_dt_obj = doc['datetime']
        if current_dt_obj not in datetime_dict:
            continue

        inversed_dt_obj = datetime_dict[current_dt_obj]

        # (1) Open <-> Close 값 교환 (Swap)
        original_open = doc['open']
        original_close = doc['close']

        new_open = original_close
        new_close = original_open
    
        # (2) 날짜/시간 포맷팅
        date_ = inversed_dt_obj.strftime("%Y%m%d")
        time_ = inversed_dt_obj.strftime("%H%M")
        
        # 새로 생성한 ID
        new_id = f"INVERSED_BTCUSDT_P_5M_{date_}_{time_}"

        # (3) 업데이트할 데이터 셋 구성
        update_doc = {
            "datetime": inversed_dt_obj,
            "open": new_open,
            "close": new_close,
            "high": doc['high'],
            "low": doc['low'],
            "volume": doc['volume'],
            "sum_open_interest_value": doc['sum_open_interest_value']
        }

        # [수정 3] UpdateOne 로직 수정
        # 필터: doc['_id']는 존재하지도 않고(Projection 0), 타겟 DB엔 new_id로 저장해야 함.
        # upsert=True: 타겟 컬렉션에 데이터가 없으면 Insert, 있으면 Update
        operations.append(
            UpdateOne({"_id": new_id}, {"$set": update_doc}, upsert=True)
        )
        
        # 메모리 관리를 위해 2000~10000개 단위로 실행
        if len(operations) >= 10000:
            target_collection.bulk_write(operations, ordered=False)
            operations = []
    
    # 남은 데이터 처리
    if operations:
        target_collection.bulk_write(operations, ordered=False)
        print("모든 작업 완료")