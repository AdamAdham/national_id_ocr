from app.utils import detect_and_process_id_card

ids_path = "app/test_ids/"
print(detect_and_process_id_card(ids_path + "youssef.jpg"))
# print(detect_and_process_id_card(ids_path + "incorrect_id.jpg"))
