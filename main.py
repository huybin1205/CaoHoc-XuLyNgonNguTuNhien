from pyvi import ViTokenizer, ViPosTagger, ViUtils
if __name__ == '__main__':
    content = u'Ngày 04/11/2021 - Giờ thi: 15:45: Giấy làm bài của sinh viên Huỳnh Khang Duy (MSSV: 2082000715), do sinh viên tự viết, không giống với mẫu giấy thi của Phòng Đào tạo - Khảo thí. Trang đầu tiên không có kẻ khung cho giám thị và giám khảo ký tên và bài làm không ghi số trang. Khi được CBCT nhắc nhở thì sinh viên Huỳnh Khang Duy thoát ra khỏi Google Meet và không có bất kỳ liên lạc hoặc động thái chỉnh sửa gì cả.'
    content = ViPosTagger.postagging(ViTokenizer.tokenize(content))
    arr = str(content).split('.')
    print(content)

    # print(b)