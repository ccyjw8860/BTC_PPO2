from db.dbclass import DBClass
from datetime import datetime, timedelta
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
    # end_str = "2022-06-01T23:55:00.000+00:00"

    db = DBClass()
    input_collection = db.BTCUSDTP_5Min_Aggregated
    target_collection = db.INVERSED_BTCUSDTP_5Min_Aggregated
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
        "buy_amount":0,
        "date":0,
        "sell_amount":0
    }

    operations = []
    count = 0
    cursor = input_collection.find(query, projection)
    for doc in tqdm(cursor):
        update_doc = {}
        current_dt_obj = doc['datetime']
        if current_dt_obj not in datetime_dict:
            continue

        inversed_dt_obj = datetime_dict[current_dt_obj]
        date_ = inversed_dt_obj.strftime("%Y%m%d")
        time_ = inversed_dt_obj.strftime("%H%M")
        update_doc['date'] = int(date_)
        update_doc['datetime'] = inversed_dt_obj
        new_id = f"INVERSED_BTCUSDT_P_5M_{date_}_{time_}"
        # (5) [핵심] 히스토그램 변환 로직 (Buy <-> Sell, 0 <-> 9)
        for i in range(10):
            target_idx = 9 - i  # 0->9, 1->8, ... 9->0
            
            # Key 이름 생성
            src_buy_shape = f"buy_hist_shape_{i}"
            src_sell_shape = f"sell_hist_shape_{i}"
            src_buy_strength = f"buy_hist_strength_{i}"
            src_sell_strength = f"sell_hist_strength_{i}"

            tgt_buy_shape = f"buy_hist_shape_{target_idx}"
            tgt_sell_shape = f"sell_hist_shape_{target_idx}"
            tgt_buy_strength = f"buy_hist_strength_{target_idx}"
            tgt_sell_strength = f"sell_hist_strength_{target_idx}"

            # Shape 변환
            if src_buy_shape in doc:
                update_doc[tgt_sell_shape] = doc[src_buy_shape]
            if src_sell_shape in doc:
                update_doc[tgt_buy_shape] = doc[src_sell_shape]

            # Strength 변환
            if src_buy_strength in doc:
                update_doc[tgt_sell_strength] = doc[src_buy_strength]
            if src_sell_strength in doc:
                update_doc[tgt_buy_strength] = doc[src_sell_strength]

        operations.append(
            UpdateOne({"_id": new_id}, {"$set": update_doc}, upsert=True)
        )
        count += 1
        if count >= 10000:
            target_collection.bulk_write(operations, ordered=False)
            operations = []
            count = 0

    if operations:
        target_collection.bulk_write(operations, ordered=False)
        print("모든 작업 완료")