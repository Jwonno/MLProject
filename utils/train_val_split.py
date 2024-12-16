import pandas as pd
import os

# Read the label data
def train_val_split(txt_file_path, output_base_dir='./dataset/stanford_products'):
    # 레이블 파일 읽기
    df = pd.read_csv(txt_file_path, sep=' ', header=0)
    
    # class_id별 이미지 수 확인
    class_counts = df['class_id'].value_counts()
    
    # 6개 이상의 이미지를 가진 class_id 필터링
    valid_classes = class_counts[class_counts >= 6].index
    
    df['valid'] = False
    
    for class_id in valid_classes:    
        df.loc[df['class_id'] == class_id, 'valid'] = True
    
    filtered_df = df[df['valid'] == True]
    
    val_df = filtered_df.groupby('class_id').sample(n=2, random_state=42)
    train_df = df.drop(val_df.index)
        
    train_df = train_df.drop(labels='valid', axis=1)
    val_df = val_df.drop(labels='valid', axis=1)
    
    
    train_path = os.path.join(output_base_dir, 'Ebay_train_split.txt')
    val_path = os.path.join(output_base_dir, 'Ebay_val_split.txt')
    
    # 분할된 데이터셋 저장
    train_df.to_csv(train_path, sep=' ', index=False, header=['image_id', 'class_id', 'super_class_id', 'path'])
    val_df.to_csv(val_path, sep=' ', index=False, header=['image_id', 'class_id', 'super_class_id', 'path'])
    
    print(f"Train images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    
    
def query_db_split(txt_file_path) -> None:
    
    query_path = txt_file_path.split('_')[:-1]
    db_path = txt_file_path.split('_')[:-1]
    
    query_path.append('query_split.txt')
    db_path.append('db_split.txt')
    
    query_path = ('_').join(query_path)
    db_path = ('_').join(db_path)
    
    df = pd.read_csv(txt_file_path, delim_whitespace=True)

    query_df = df.groupby('class_id').sample(n=1, random_state=42)

    db_df = df.drop(query_df.index)

    os.makedirs(os.path.dirname(query_path), exist_ok=True)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    query_df.to_csv(query_path, sep=' ', index=False)
    db_df.to_csv(db_path, sep=' ', index=False)
    
    print(f"Query images: {len(query_df)}")
    print(f"Database images: {len(db_df)}")
    print("Dataset split completed.")
    
if __name__ == '__main__':
    train_val_split('./dataset/stanford_products/Ebay_train.txt', './dataset/stanford_products')
    query_db_split('./dataset/stanford_products/Ebay_test.txt')
    