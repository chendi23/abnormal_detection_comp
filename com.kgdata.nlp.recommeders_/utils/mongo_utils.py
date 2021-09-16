#!/usr/bin/python3
# -*- coding: utf-8 -*-
import gridfs
import os
import struct

'''
对mongoDB操作的工具类
'''


# 插入单条记录,返回记录的_id
def insert_one_record_to_mongo(col, record):
    id = str(col.insert_one(record).inserted_id)
    return id


# 插入多条记录
def insert_records_to_mongo(col, records):
    col.insert_many(records)


# 若存在记录，则更新；若不存在，则新增记录
def save_record_to_mongo(col, record):
    id = str(col.save(record))
    return id


# 查询单条记录（通过主键)
def get_one_record_by_id(col, id):
    record = col.find_one({'_id': id})
    return record


# 查询单条记录（通过条件）。condition为字典类型
def get_one_record_by_condition(col, condition):
    record = col.find_one(condition)
    return record


# 查询多条记录（通过条件）。condition为字典类型
def get_records_by_condition(col, condition):
    records = col.find(condition)
    return records


# 更新指定记录（通过主键）
def update_one_record_by_id(col, id, updated_parameters):
    condition = {'_id': id}
    return col.update_one(condition, {'$set': updated_parameters})


# 更新指定记录（通过条件）
def update_one_record_by_condition(col, condition, updated_parameters):
    return col.update_one(condition, {'$set': updated_parameters})


# 更新多条记录（通过条件）
def update_records_by_condition(col, condition, updated_parameters):
    return col.update(condition, {'$set': updated_parameters})


# 删除指定记录（通过主键）
def delete_one_record_by_id(col, id):
    condition = {'_id': id}
    col.delete_one(condition)


# 删除指定记录（通过条件）
def delete_one_record_by_condition(col, condition):
    return col.delete_one(condition)


# 删除多条记录（通过条件）
def delete_records_by_condition(col, condition):
    return col.delete_many(condition)


# 判断文件是否存在
def check_file_existed(db, col_name, condition):
    fs = gridfs.GridFS(db, col_name)
    grid_out = fs.find_one(condition)
    if not grid_out:
        return False
    return True


# 读取单个文件的内容，并转化为utf-8编码格式
def read_one_file_data_from_mongo(db, col_name, condition):
    fs = gridfs.GridFS(db, col_name)
    grid_out = fs.find_one(condition)
    data = grid_out.read().decode('utf-8')
    return data


# 读取多个文件的内容，转化为utf-8编码格式，并合并
def read_files_data_from_mongo(db, col_name, condition):
    fs = gridfs.GridFS(db, col_name)
    data = ''
    for grid_out in fs.find(condition):
        data += grid_out.read().decode('utf-8')
    return data


# 保存文件到mongo。f为文件流
def save_file_to_mongo(db, col_name, f, **kwargs):
    data = f.read()
    fs = gridfs.GridFS(db, col_name)
    return str(fs.put(data, **kwargs))


# 删除文件（根据条件）
def delete_files_from_mongo(db, col_name, condition):
    fs = gridfs.GridFS(db, col_name)
    for grid_out in fs.find(condition):
        fs.delete(grid_out._id)


# 读取mongo单个文件的内容，并存到本地
def read_one_file_from_mongo_and_save_local(db, col_name, condition, model_path):
    # 若路径所在文件夹不存在，则创建文件夹
    model_dir = os.path.split(model_path)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    fs = gridfs.GridFS(db, col_name)
    grid_out = fs.find_one(condition)
    data = grid_out.read()
    with open(model_path, 'wb') as f:
        for line in data:
            f.write(struct.pack('B', line))


# 读取mongo多个文件的内容，并存到本地
def read_files_from_mongo_and_save_local(db, col_name, condition, model_dir):
    # 若模型文件夹不存在，则创建文件夹
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    fs = gridfs.GridFS(db, col_name)
    for grid_out in fs.find(condition):
        data = grid_out.read()
        write_path = os.path.join(model_dir, grid_out._file['fileName'])
        with open(write_path, 'wb') as f:
            for line in data:
                f.write(struct.pack('B', line))


# 加入排序、限制数量进行组合查询
def get_records_by_sort_limit(col, condition, sort_condition, limit_num):
    records = col.find().sort(sort_condition).limit(limit_num)
    return records
