from flask import Flask, request, render_template, jsonify, send_file, abort
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import logging
import kaleido
import base64
import pyodbc
import json
import io
import os
import re

app = Flask(__name__)

# SQL Server 連接配置
server = '10.30.163.208'
username = 'sa'
password = 'Auol6bi1'
database = 'ArrayPH'

connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

@app.route('/')
def index():
    return render_template('index.html', table_html='')

data_draw_CD_name = None  # 定義全局變數
data_draw_DCD = None

@app.route('/details/<glass_id>', methods=['GET', 'POST'])
def details(glass_id):
    global data_draw_CD_name
    # 從 URL 中獲取 `me_date`
    me_date = request.args.get('me_date')
    
    # 如果沒有從 URL 中獲取 `me_date`，則查詢 AR_Titan
    if not me_date:
        query_titan = f'''
            SELECT Recipe_ID, ME_DATE 
            FROM dbo.AR_Titan 
            WHERE Glass_ID = '{glass_id}'
        '''
        try:
            with pyodbc.connect(connection_string) as conn:
                logging.debug(f"Connecting to database with connection string: {connection_string}")
                df_titan = pd.read_sql(query_titan, conn)

            if df_titan.empty:
                logging.warning(f"No Recipe_ID or ME_DATE found for Glass ID: {glass_id} in AR_Titan.")
                return jsonify({"error": "No Recipe_ID or ME_DATE found in AR_Titan."}), 404

            recipe_id = df_titan.iloc[0]['Recipe_ID']
            me_date = df_titan.iloc[0]['ME_DATE']  # 從資料庫獲取 ME_DATE
        except Exception as e:
            logging.error(f"Error querying AR_Titan: {e}")
            return jsonify({"error": "Database query error for AR_Titan"}), 500
    else:
        # 如果從 URL 獲得了 `me_date`，則使用該值
        try:
            with pyodbc.connect(connection_string) as conn:
                #用ME_DATE找尋當時的Recipe_ID
                query_titan = f'''
                     SELECT Recipe_ID
                     FROM dbo.AR_Titan
                     WHERE Glass_ID = '{glass_id}' AND CONVERT(varchar, ME_DATE, 120) = '{me_date[:19]}'
                 '''
#                 query_titan = f'''
#                     SELECT Recipe_ID
#                     FROM dbo.AR_Titan
#                     WHERE Glass_ID = '{glass_id}'
#                 '''
                df_titan = pd.read_sql(query_titan, conn)
                if df_titan.empty:
                    return jsonify({"error": "No Recipe_ID found in AR_Titan for the specified Glass_ID."}), 404
                
                recipe_id = df_titan.iloc[0]['Recipe_ID']
        except Exception as e:
            logging.error(f"Error querying AR_Titan for Recipe_ID: {e}")
            return jsonify({"error": "Database query error for AR_Titan"}), 500

    # 從 Recipe_ID 提取數字部分
    match = re.search(r'(\d+)$', recipe_id)
    if not match:
        logging.warning(f"Invalid Recipe_ID format: {recipe_id} for Glass ID: {glass_id}")
        return jsonify({"error": "Invalid Recipe_ID format."}), 404

    recipe_id_numbers = match.group(1)
#     print(recipe_id_numbers)

    # 在執行 query_measure 查詢之前，記錄 Glass_ID、Recipe_ID 和 ME_DATE 的值
    logging.debug(f"Glass_ID: {glass_id}, Recipe_ID: {recipe_id_numbers}, ME_DATE: {me_date}")

    # 使用 URL 中的 me_date 或從 AR_Titan 獲取的 me_date 查詢 AR_Measure
#     query_measure = f'''
#         SELECT * 
#         FROM dbo.AR_Measure 
#         WHERE Glass_ID = '{glass_id}' AND Recipe_ID LIKE '%{recipe_id_numbers}%' AND Test_Time = '{me_date}'
#     '''
    
    query_measure = f'''
        SELECT * 
        FROM dbo.AR_Measure 
        WHERE Glass_ID = '{glass_id}' 
        AND Recipe_ID LIKE '%{recipe_id_numbers}%' 
        AND CONVERT(varchar, Test_Time, 120) LIKE '{me_date[:16]}%'
        
    '''
    
    logging.debug(f"Executing query for AR_Measure: {query_measure}")

    try:
        with pyodbc.connect(connection_string) as conn:
            df_measure = pd.read_sql(query_measure, conn)

        if df_measure.empty:
            logging.warning(f"No data found in AR_Measure for Glass ID: {glass_id}, Recipe_ID: {recipe_id}, and ME_DATE: {me_date}")
            return jsonify({"error": "No data found in AR_Measure for this Glass ID, Recipe_ID, and ME_DATE."}), 404

#         # 只挑選每個 Point_No 的第一筆資料
#         df_measure = df_measure.groupby('Point_No').nth(0).reset_index()
#         print(df_measure.columns)

        # # 將索引轉換為 Point_No 欄位
        df_measure['Point_No'] = df_measure.index + 1

        # 如果需要重設索引，可以這樣做
        df_measure.reset_index(drop=True, inplace=True)

#         # 打印列名以確認變更
#         print(df_measure.columns)

        # 替換 L1~G2 列中的 None 和 NaN 為空字串
        columns_to_replace = ['L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2', 'G1', 'G2']
        df_measure[columns_to_replace] = df_measure[columns_to_replace].replace({None: '', np.nan: ''})
        
        
        #漏點不刪除，不然影響Scan的Group
#         df_filtered = df_measure[~(df_measure[columns_to_replace].isnull() | df_measure[columns_to_replace].eq('')).all(axis=1)]

        selected_columns = [
            'Point_No', 'X_R', 'Y_R', 'PointJudge', 'SubRecipeNo', 
            'L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2', 
            'G1', 'G2', 'Point_Chip', 'Line_Name', 'Scan', 
            'Lens', 'SPEC', 'special_type', 'Img1', 'Img2'
        ]
        filtered_df = df_measure[selected_columns]
        filtered_df_sort = filtered_df.sort_values(by='Point_No', ascending=True)
#         print(filtered_df.columns)

        # 將 Img1 和 Img2 轉換為 IMG 標籤
        filtered_df_sort['Img1'] = filtered_df_sort['Img1'].apply(lambda x: f'<img src="{x}" width="100" height="100" />' if pd.notnull(x) and x != '' else '')
        filtered_df_sort['Img2'] = filtered_df_sort['Img2'].apply(lambda x: f'<img src="{x}" width="100" height="100" />' if pd.notnull(x) and x != '' else '')

        # 使用第一行的數據作為顯示（若有需要的話）
        data = df_measure.iloc[0].to_dict()
        
        table_html = filtered_df_sort.to_html(classes='table table-striped table-bordered', index=False, escape = False)#escape = False 是在Img1和2印出圖

########################################################################################################################     

# #統計量表格

        #all
        filtered_df_sort = pd.DataFrame(filtered_df_sort)
        
        # 創建新的 DataFrame，這裡正確的方式是使用雙重方括號
        filtered_df_sort_stat = filtered_df_sort[['L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2']]
#         print(filtered_df_sort_stat)
        # 處理缺失值
        filtered_df_sort_stat.replace('', pd.NA, inplace=True)
        filtered_df_sort_stat.fillna(filtered_df_sort_stat.mean(), inplace=True)
#         print(filtered_df_sort_stat)
        
        # 計算統計數據
        stats_all = filtered_df_sort_stat.agg(['mean', 'std', 'max', 'min']).T
        stats_all = stats_all.round(2)
        stats_all.fillna('', inplace = True)
        
        statistics = stats_all.to_dict(orient='index')  # 將統計數據轉換為字典格式

        # 輸出統計數據
#         print(statistics)

# ########################################################################################################################

# # 計算每個Scan的統計量
        def calculate_stats_scan(df):
            scan_dataframes = {}
            stats_summary = {}

            for scan_name, group in df.groupby('Scan'):
                scan_df_name = f'{scan_name}'
                scan_dataframes[scan_df_name] = group.copy()

        # 確保目標欄位為數字型別，無法轉換的設置為 NaN
                target_columns = [col for col in ['L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2'] if col in group.columns]
                scan_dataframes[scan_df_name][target_columns] = scan_dataframes[scan_df_name][target_columns].apply(
                    pd.to_numeric, errors='coerce'
                )

        # 填充 NaN 值為均值
                scan_dataframes[scan_df_name][target_columns] = scan_dataframes[scan_df_name][target_columns].apply(
                    lambda x: x.fillna(x.mean())
                )

        # 計算統計量
                stats = scan_dataframes[scan_df_name][target_columns].agg(['mean', 'std', 'max', 'min']).T.round(2)
                stats_summary[scan_df_name] = stats.replace(np.nan, '')  # 替換 NaN 為空字串

            return stats_summary

# 計算統計摘要
        stats_summary = calculate_stats_scan(filtered_df_sort)
#         print(stats_summary)



########################################################################################################################
#散布圖
        try:
             # 假設 df_measure 是你的數據框
             # 將 X_R 和 Y_R 列轉換為數字類型，如果無法轉換則設為 NaN
            df_measure['X_R'] = pd.to_numeric(df_measure['X_R'], errors='coerce')
            df_measure['Y_R'] = pd.to_numeric(df_measure['Y_R'], errors='coerce')

            # 移除任何含有 NaN 值的行，這樣可以確保數據都是有效數字
            df_measure = df_measure.dropna(subset=['X_R', 'Y_R'])

            # 將 X_R 和 Y_R 進位至小數點第一位
            df_measure['X_R'] = df_measure['X_R'] #.round(1)
            df_measure['Y_R'] = df_measure['Y_R'] 

            # 使用 Plotly 繪製交互式散佈圖
            fig_plotly = px.scatter(
                df_measure,
                x='X_R', 
                y='Y_R', 
                hover_data=['Point_No', 'X_R', 'Y_R'],  # 顯示 Point_No, X_R, Y_R
                title="Interactive Scatter Plot for X(R) and Y(R)",
                opacity=0.5  # 增加透明度
            )

            # 設置 Plotly 圖的軸標題和格式
            fig_plotly.update_layout(
                xaxis_title='X(R)',
                yaxis_title='Y(R)',
                title_font_size=26,
                xaxis_title_font_size=18,
                yaxis_title_font_size=18,
                xaxis=dict(tickmode='linear', dtick=100),
                yaxis=dict(tickmode='linear', dtick=100),
                width=900,  # 設置您希望的寬度
                height=500,  # 設置您希望的高度
                margin=dict(l=50, r=50, t=50, b=50)  # 設定邊距
            )
        
#             fig_plotly.write_html('scatter_plot.html', include_plotlyjs='cdn')
            plotly_html_content = pio.to_html(fig_plotly, full_html=False, include_plotlyjs='cdn')
    
        except Exception as e:
            logging.error(f"Error while creating scatter plot: {e}")

########################################################################################################################     
# filtered_df若有欄位是''，以中位數填補它

        try:
                # 要填補的欄位列表
                columns_to_fill = ['L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2']

                # 將每個指定的欄位中的空字串轉換為 NaN
                for column in columns_to_fill:
                    filtered_df[column] = pd.to_numeric(filtered_df[column], errors='coerce')

                # 填補中位數或預設值
                default_value = 0  # 當整個欄位都為空時使用的預設值

                for column in columns_to_fill:
                    if filtered_df[column].isnull().all():  # 檢查整個欄位是否都為 NaN
                        filtered_df[column].fillna(default_value, inplace=True)  # 用預設值填補
                    else:
                        median_value = filtered_df[column].median()  # 計算中位數
                        filtered_df[column].fillna(median_value, inplace=True)  # 用中位數填補遺失值
            
#       整理資料，創建新欄位Group，命名左、左中、中、右中、右
                filtered_df = filtered_df.to_numpy()

                data_draw_name = assign_group(filtered_df)
#                 print(data_draw_name)
        
                data_draw_CD = process_data(filtered_df,data_draw_name, glass_id)
#                 print(data_draw_CD)

                rows = []

        # 遍歷 data_draw_CD 的每個子列表
                for group in data_draw_CD:
                    for entry in group:
                # 檢查 entry 是否為有效列表，並且包含至少 9 個元素
                        if isinstance(entry, list) and len(entry) == 9:
                            Scan_M = entry[0]  # 提取 Scan_M 字符串
                            values = entry[1:8]  # 提取對應的數值 (L1, L2, L3, L4, L5, A1, A2)
                            group_label = entry[8]  # 提取組標籤

                    # 跳過空的 Scan_M 或 None 值
                            if Scan_M and any(val is not None for val in values):  # 確保至少有一個值不為 None
                        # 去除 Scan_M 中的多餘空格
                                Scan_M_clean = re.sub(r'\s+', '', Scan_M)
                        # 添加到 rows 列表中
                                rows.append([Scan_M_clean] + values + [group_label])  # 包含組標籤
                        else:
                    # 只對無效條目發出一次警告
                            if entry != ['', None]:  # 僅排除特定無效條目
                                print(f"Warning: 'entry' is not a valid list or has insufficient elements. Entry: {entry}")

        # 將結果轉換成 DataFrame
                columns = ['Scan_M', 'L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2', 'Group']  # 增加所有欄位
                data_draw_CD_name = pd.DataFrame(rows, columns=columns)
        
                columns_to_replace = ['L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2']
                data_draw_CD_name[columns_to_replace] = data_draw_CD_name[columns_to_replace].replace({None: '',np.nan: ''})# 替換 L1~G2 列中的 None 和 NaN 為空字串
#                 print(data_draw_CD_name['Group'].unique())
                
        #依照Group和Scan_M排序
        
        # 定義 Group 的排序順序
                group_order = ['Left', 'Left-Center', 'Center', 'Right-Center', 'Right', 'Left-Outer','Left-Inner','Right-Inner','Right-Outer']
            
        #檢查是否在data的欄位中有這些欄位
                valid_groups = [group for group in group_order if group in data_draw_CD_name['Group'].unique()]
            
                data_draw_CD_name['Group'] = pd.Categorical(data_draw_CD_name['Group'], categories=valid_groups, ordered=True)

                # 提取出 Sort_Scan 和 Sort_M，並應用排序邏輯
                data_draw_CD_name['Sort_Scan'] = data_draw_CD_name['Scan_M'].apply(sort_scan)

                data_draw_CD_name['Sort_M'] = data_draw_CD_name['Scan_M'].apply(sort_m_o)

                # 按照 Group 和 Sort_Scan、Sort_M 列進行排序
                data_draw_CD_name = data_draw_CD_name.sort_values(by=['Group', 'Sort_Scan', 'Sort_M']).reset_index(drop=True)

        # 檢查有效欄位，按照 Group 和 Sort_Scan、Sort_M 進行排序
                valid_groups = data_draw_CD_name['Group'].unique()
                data_draw_CD_name['Group'] = pd.Categorical(data_draw_CD_name['Group'], categories = valid_groups, ordered = True)
                data_draw_CD_name = data_draw_CD_name.sort_values(by=['Group', 'Sort_Scan', 'Sort_M']).reset_index(drop=True)

        # 將 Point_No 更新為新的排序後的索引
#                 data_draw_CD_name['Point_No'] = range(1, len(data_draw_CD_name) + 1)
#                 print(data_draw_CD_name)

                data_draw_DCD = generate_DCD_data(data_draw_CD_name)
                
#                 print(data_draw_DCD['Diff_Scan_M'].unique())
                
        except Exception as e:
            logging.error(f"Error while data_draw: {e}")

########################################################################################################################

# # CD anf DCD Chart for details
        try:
                default_column = 'L1'
                all_groups = data_draw_CD_name['Group'].unique().tolist()
                cd_plot_html = generate_CD_plot(default_column, all_groups)

                #DCD plot
                dcd_plot_html = generate_DCD_plot(f'Diff-{default_column}', all_groups)
                
        except Exception as e:
            logging.error(f"Error while generating plots: {e}")
########################################################################################################################
        # 將散布圖嵌入HTML
#         with open('scatter_plot.html', 'r', encoding='utf-8') as f:
#             plotly_html_content = f.read()

#         return render_template('details.html',draw_columns=draw_columns ,selected_chart=selected_chart , table_html=table_html, html_content=plotly_html_content, glass_id=glass_id, recipe_id=recipe_id, data=data)
        return render_template('details.html',
                               draw_columns=['L1','L2','L3','L4','L5','A1','A2'],
#                                selected_chart=selected_chart,
                               table_html=table_html,
                               statistics=statistics,
                               stats_summary = stats_summary,
                               html_content=plotly_html_content,
                               cd_plot_html=cd_plot_html,
                               dcd_plot_html = dcd_plot_html,
                               groups=all_groups, 
                               selected_column=default_column, 
                               selected_groups=all_groups,
                               glass_id=glass_id,
                               recipe_id=recipe_id,
                               data=data
                               )

    except Exception as e:
        logging.error(f"Error while querying the database: {e}")
        return jsonify({"error": str(e)}), 500  # 返回具體的錯誤信息


@app.route('/search', methods=['GET'])
def search():
    # 取得查詢參數
#     global recipe_id_numbers
    
    start_time = request.args.get('st', '')
    end_time = request.args.get('et', '')
    lot_id = request.args.get('lotid', '')
    recipe = request.args.get('Recipe', '')

    # 構建初始 SQL 查詢語句，Lot_ID 不能為空,使用ME_DATE作為顯示時間
    query = '''
        SELECT Lot_ID, Glass_ID, OP_No, DEV_NO, PH_STEP, Recipe_ID, PH_EQ, csv_floder, ME_DATE AS Data_Time
        FROM dbo.AR_Titan
        WHERE Lot_ID IS NOT NULL AND Lot_ID <> ''
    '''

    # 構建參數列表
    params = []

    # 檢查是否提供了起始和結束時間，若有則加入時間過濾條件
    if start_time and end_time:
        start_time = start_time.replace('T', ' ') + ':00'
        end_time = end_time.replace('T', ' ') + ':00'
        query += " AND Last_PH_TIME BETWEEN ? AND ?"
        params.extend([start_time, end_time])

    if lot_id:
        query += " AND Lot_ID LIKE ?"
        params.append('%' + lot_id + '%')

    if recipe:
        query += " AND Recipe_ID LIKE ?"
        params.append('%' + recipe + '%')

    if not (start_time and end_time) and not lot_id and not recipe:
        return jsonify({"error": "Please provide at least one search criterion: time range, Lot_ID, or Recipe_ID"}), 400

    try:
        with pyodbc.connect(connection_string) as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            return jsonify({"error": "No data found for the given criteria."}), 404
        
        df['Data_Time'] = pd.to_datetime(df['Data_Time'])
        df = df.sort_values(by='Data_Time', ascending=False)

        # 在這裡記錄 ME_DATE 以便在 details 頁面中使用
        me_dates = df['Data_Time'].tolist()  # 獲取 ME_DATE 列表
#         print(me_dates)
        table_html = df.to_html(classes='table table-striped table-bordered', index=False)

        return render_template('result.html', data=df.to_dict(orient='records'), table_html=table_html, start_date=start_time, end_date=end_time, me_dates=me_dates)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/download_csv/<filename>') #將dat轉為csv用，但目前無效(AR01/AR02才有這問題)
def down_csv(filename):
    dat_file = f'{filename}.dat'
    csv_file = f'{filename}.csv'
    
    try:
        if os.path.exists(dat_file):  # 修改這裡
            df = pd.read_csv(dat_file, sep=None, engine='python')  # 讀取.dat檔案
            
            df.to_csv(csv_file, index=False)  # 將數據儲存為.csv檔案
            
            return send_file(csv_file, as_attachment=True, download_name=f'{filename}.csv')  # 修正拼寫
            
        else:
            return jsonify({'error': f'{dat_file}不存在'}), 404
    
    except Exception as e:
        return f'轉換或下載時發生錯誤: {str(e)}', 500
    
    finally:
        if os.path.exists(csv_file):
            try:
                os.remove(csv_file)
            except Exception as e:
                print(f'刪除檔案時發生錯誤: {str(e)}')
                
########################################################################################################################
def group_rule_1(k):
    if 60 <= k[0] < 260:
        k[1] = 'Left'
    elif 260 <= k[0] < 460:
        k[1] = 'Left-Center'
    elif 460 <= k[0] < 660:
        k[1] = 'Center'
    elif 660 <= k[0] < 860:
        k[1] = 'Right-Center'
    elif 860 <= k[0] < 990:
        k[1] = 'Right'
    elif 990 <= k[0] < 1190:
        k[1] = 'Left'
    elif 1190 <= k[0] < 1390:
        k[1] = 'Left-Center'
    elif 1390 <= k[0] < 1590:
        k[1] = 'Center'
    elif 1590 <= k[0] < 1790:
        k[1] = 'Right-Center'
    elif 1790 <= k[0] < 1990:
        k[1] = 'Right'
    else:
        k[1] = 'Unknown'  # 為未知情況設置默認值

    return k[1]  # 返回分組名稱

def group_rule_2(k):
    if 0 <= k[0] < 100 or 600 <= k[0] < 700 or 1230 <= k[0] < 1250 :
        k[1] = 'Left'
    elif 150 <= k[0] < 200 or 700 <= k[0] < 800 or 1300 <= k[0] < 1400:
        k[1] = 'Left-Center'
    elif 300 <= k[0] < 400 or 900 <= k[0] < 1000 or 1500 <= k[0] < 1600:
        k[1] = 'Center'
    elif 400 <= k[0] < 500 or 1000 <= k[0] < 1100 or 1600 <= k[0] < 1700:
        k[1] = 'Right-Center'
    elif 600 <= k[0] < 700 or 1200 <= k[0] < 1230 or 1800 <= k[0] < 1900:
        k[1] = 'Right'
    else:
        k[1] = 'Unknown'  # 為未知情況設置默認值

    return k[1]  # 返回分組名稱


def assign_group(Data, rule='rule1'):
    data_draw_name = []  # 初始化一組資料集
    data_draw_name.append([round(float(Data[0][1]) / 10) * 10, Data[0][16]])

    for i in range(len(Data) - 1):
        try:
            if (
                abs(float(Data[i][1]) - float(Data[i + 1][1])) > 10 or
                Data[i][16] != Data[i + 1][16] or
                Data[i + 1][19] == 'CenterCD' or
                Data[i][19] == 'CenterCD'
            ) and Data[i + 1][1] != '':
        
                found = False
        
                for s in range(len(data_draw_name)):
                    if data_draw_name[s][0] == round(float(Data[i + 1][1]) / 10) * 10:
                        found = True
                        break

                if not found:
                    data_draw_name.append([0, ''])
                    data_draw_name[-1][0] = round(float(Data[i + 1][1]) / 10) * 10
                    data_draw_name[-1][1] = Data[i + 1][16]

        except IndexError:
            print(f"Skipping out-of-range data at index {i + 1}")
            continue

    data_draw_name.sort(key=lambda x: x[0])  # 根據第一個元素排序

    # 分配群組
    for k in data_draw_name:
        if rule == 'rule1':
            k[1] = group_rule_1(k)
        elif rule == 'rule2':
            k[1] = group_rule_2(k)

    # 處理 Unknown 的部分
    for idx, k in enumerate(data_draw_name):
        if k[1] == 'Unknown':  # 檢查是否為 Unknown
            # 尋找最近的非 Unknown 群組
            closest_group = None
            min_distance = float('inf')

            # 往前尋找最近的群組
            for i in range(idx - 1, -1, -1):
                if data_draw_name[i][1] != 'Unknown':
                    distance = abs(k[0] - data_draw_name[i][0])
                    if distance < min_distance:
                        min_distance = distance
                        closest_group = data_draw_name[i][1]
                    break  # 找到最近群組後停止

            # 往後尋找最近的群組
            for i in range(idx + 1, len(data_draw_name)):
                if data_draw_name[i][1] != 'Unknown':
                    distance = abs(k[0] - data_draw_name[i][0])
                    if distance < min_distance:
                        min_distance = distance
                        closest_group = data_draw_name[i][1]
                    break  # 找到最近群組後停止

            # 將 Unknown 值歸類到最近的群組
            if closest_group:
                k[1] = closest_group
                print(f"Assigning Unknown value at {k[0]} to nearest group: {closest_group}")

    return data_draw_name


# 定義處理 'Scan' 和 'S1/2' 的排序函數
def sort_scan(scan_value):
    if 'Scan' in scan_value:
        try:
            return int(scan_value.split('Scan')[1].split('_')[0])  # 提取 'Scan' 的數字部分
        except (ValueError, IndexError):
            return float('inf')  # 解析失敗的話，排在最後
    elif 'S' in scan_value and '/' in scan_value:
        try:
            # 對於 'S1/2' 類型的值，返回一個介於 Scan1 和 Scan2 之間的值 (如 1.5)
            range_values = scan_value.split('_')[0].split('S')[1].split('/')
            return (float(range_values[0]) + float(range_values[1])) / 2
        except (ValueError, IndexError):
            return float('inf')  # 解析失敗的話，排在最後
    else:
        return float('inf')


# 定義處理 'M' 和 'O' 的排序函數
def sort_m_o(scan_m_value):
    if 'M' in scan_m_value:
        try:
            return float(scan_m_value.split('M')[1].split('->')[0])  # 提取 'M' 之後的數字部分
        except (ValueError, IndexError):
            return float('inf')  # 解析失敗的話，排在最後
    elif 'O' in scan_m_value:
        try:
            # 對於 'O1/2' 類型的值，返回一個介於 M1 和 M2 之間的值 (如 1.5)
            m_values = scan_m_value.split('O')[1].split('->')[0]
            range_m_values = m_values.split('/')
            return (float(range_m_values[0]) + float(range_m_values[1])) / 2
        except (ValueError, IndexError):
            return float('inf')  # 解析失敗的話，排在最後
    else:
        return float('inf')

    
def process_data(Data, data_draw_name, glass_id):
    tt = 0
    data_draw_CD = []

    # 初始化 data_draw_CD 為空列表
    for j in range(7):  # 假設要處理的列是 5 到 11，共 7 列
        data_draw_CD.append([])

    # 遍歷 Data 中的每一行
    for i in range(len(Data)):
        # 檢查 Data 的第 19 列是否不是 'CenterCD'
        if Data[i, 19] != 'CenterCD':
            for s in range(len(data_draw_name)):  # 使用 data_draw_name 的長度
                if (data_draw_name[s][0] > float(Data[i, 1]) - 7.6) and \
                   (data_draw_name[s][0] < float(Data[i, 1]) + 7.6):

                    # 65" 特定條件處理 (跨 scan 比較)
                    m = len(data_draw_CD[j])
                    if (glass_id[0] != 'F' and (m > 0 and data_draw_CD[j][m - 1][0][:5] != Data[i, 16])) or \
                       (glass_id[0] == 'F' and (i + 1 < len(Data) and Data[i + 1, 14] != Data[i, 14])):
                        # 插入空格
                        data_draw_CD[j].append(['', None, None, None, None, None, None, None, None])  # 需要有 9 個元素

                    # 抓 Y 軸資料並儲存
                    m = len(data_draw_CD[j])
                    data_draw_CD[j].append([None] * 9)  # 初始化一個長度為 9 的列表

                    # X 軸資料：組合 Data[i][16] 和 Data[i][17] 作為標籤
                    data_draw_CD[j][m][0] = f"{Data[i, 16]}_{Data[i, 17]}->{tt}"

                    # 限制 Data[i][5 + j] 到第 5~11 列 (L1~A2)
                    for k in range(7):  # L1 (5) 到 A2 (11)
                        if 0 <= j <= 6 and (5 + j) < Data.shape[1] and Data[i, 5 + k] != '':
                            data_draw_CD[j][m][k + 1] = float(Data[i, 5 + k])  # k + 1 對應 L1, L2, ..., A2
                        else:
                            data_draw_CD[j][m][k + 1] = None  # 設定為 None
                            
                    data_draw_CD[j][m][-1] = data_draw_name[s][1]# 添加所屬組的名稱到最後一個位置 (第9個元素)

                    # 跳出內部迴圈
                    break

            # 如果當前行的第 5 到第 11 列全為空，則忽略該列
            if np.all(Data[i, 5:12] == ''):
                continue

        # 如果下一行資料的 Scan（Data[i][16] 或 Data[i][17]）不同，重置 tt，否則遞增
        if i + 1 < len(Data) and (Data[i, 16] != Data[i + 1, 16] or Data[i, 17] != Data[i + 1, 17]):
            tt = 0
        else:
            tt += 1

    return data_draw_CD


@app.route('/select_column', methods=['POST'])
def select_column():
    global data_draw_CD_name
    selected_column = request.form.get('column')
    
    # 獲取選擇的 Group
    selected_groups = request.form.getlist('groups')
    
    # 根據用戶選擇生成 CD 和 DCD 圖表
    cd_plot_html = generate_CD_plot(selected_column, selected_groups)
    dcd_plot_html = generate_DCD_plot(f'Diff-{selected_column}', selected_groups)  # 使用選擇的列生成 DCD 圖表
    
    all_groups = data_draw_CD_name['Group'].unique().tolist()
    
    # 渲染模板，返回需要更新的 HTML 內容
    return render_template('chart.html', 
                           cd_plot_html=cd_plot_html,
                           dcd_plot_html=dcd_plot_html,
                           draw_columns=['L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2'], 
                           groups=all_groups, 
                           selected_column=selected_column, 
                           selected_groups=selected_groups, 
                           data=data_draw_CD_name)


def generate_CD_plot(column, selected_groups):
    global data_draw_CD_name

    if data_draw_CD_name is None or data_draw_CD_name.empty:
        return "Data not available!"

    if column not in data_draw_CD_name.columns:
        return "Column not found!"
    
    data_draw_CD_name = data_draw_CD_name.sort_values(by=['Sort_Scan', 'Sort_M'])

    grouped = data_draw_CD_name.groupby('Group')

    lines = []
    try:
        overall_mean = data_draw_CD_name[column].mean()
        std_dev = data_draw_CD_name[column].std()
        
        UCL = overall_mean + 3 * std_dev
        LCL = overall_mean - 3 * std_dev
        
        group_colors = {
        'Left': 'rgba(0, 0, 255, 0.5)',  # 半透明的藍色
        'Left-Center': 'rgba(0, 255, 0, 0.5)',  # 半透明的綠色
        'Center': 'rgba(255, 0, 0, 0.5)',  # 半透明的紅色
        'Right-Center': 'rgba(255, 165, 0, 0.5)',  # 半透明的橙色
        'Right': 'rgba(128, 0, 128, 0.5)',  # 半透明的紫色
        'Left-Outer': 'rgba(0, 255, 255, 0.5)',  # 半透明的青色
        'Left-Inner': 'rgba(255, 0, 255, 0.5)',  # 半透明的洋紅色
        'Right-Inner': 'rgba(255, 255, 0, 0.5)',  # 半透明的黃色
        'Right-Outer': 'rgba(165, 42, 42, 0.5)'  # 半透明的棕色
    }

        for group_name, group_data in grouped:
            if group_name in selected_groups:
                lines.append(go.Scatter(
                    x=group_data['Scan_M'],
                    y=group_data[column],
                    mode='lines+markers',
                    name=f'{group_name}',
                    hovertemplate=(
#                         'Point No: %{customdata[0]}<br>' +
                        'Scan_M: %{x}<br>' +
                        column + ': %{y}<extra></extra>'
                    ),
#                     customdata=group_data[['Point_No']].values,
                    line=dict(color=group_colors.get(group_name, 'rgba(0, 0, 0, 0.5)'))  # 默認為半透明的黑色
                ))

        lines.append(go.Scatter(
            x=data_draw_CD_name['Scan_M'],
            y=[overall_mean] * len(data_draw_CD_name['Scan_M']),
            mode='lines',
            name='Mean',
            line=dict(color='green', width=2, dash='dash'),
        ))

        lines.append(go.Scatter(
            x=data_draw_CD_name['Scan_M'],
            y=[UCL] * len(data_draw_CD_name['Scan_M']),
            mode='lines',
            name='UCL (Upper Control Limit)',
            line=dict(color='red', width=2, dash='dash'),
        ))

        lines.append(go.Scatter(
            x=data_draw_CD_name['Scan_M'],
            y=[LCL] * len(data_draw_CD_name['Scan_M']),
            mode='lines',
            name='LCL (Lower Control Limit)',
            line=dict(color='red', width=2, dash='dash'),
        ))

        layout = go.Layout(
            title=f'Line Plot of {column}',
            xaxis=dict(
                title='Scan_M',
                categoryorder='array',
                categoryarray=data_draw_CD_name['Scan_M'],
                tickangle = -60
            ),
            yaxis=dict(title=column),
            hovermode='closest'
        )

        fig = go.Figure(data=lines, layout=layout)

        return pio.to_html(fig, full_html=False)
    
    except Exception as e:
        return f"Error generating plot: {str(e)}"




# 自訂排序
def custom_sort_key(value):
    parts = value.split('->')
    
    def parse_part(part):
        # 處理 S1/2 類似的情況
        if 'S' in part and '/' in part:
            return (1.5,)  # S1/2 放在 Scan1 和 Scan2 之間
        scan_match = re.search(r'Scan(\d+)', part)
        if scan_match:
            return (int(scan_match.group(1)),)  # Scan後面的數字排序
        
        # 處理 M 和 O 類似的情況
        if 'M' in part:
            return (10 + int(part.split('M')[1]),)  # M 後面的數字排序
        elif 'O' in part:
            o_values = part.split('O')[1].split('/')
            return (10 + (float(o_values[0]) + float(o_values[1])) / 2,)  # O1/2 放在M1和M2之間
        
        return (float('inf'),)  # 未知的部分放在最後

    left_key = parse_part(parts[0])
    right_key = parse_part(parts[1]) if len(parts) > 1 else (float('inf'),)
    
    return left_key + right_key

# 排序結果數據框中的 Diff_Scan_M 欄位
def generate_DCD_data(data_draw_CD_name):
    global data_draw_DCD

    # 檢查是否有 'Group' 和 'Scan_M' 欄位
    if 'Group' not in data_draw_CD_name.columns or 'Scan_M' not in data_draw_CD_name.columns:
        return "Data must contain 'Group' and 'Scan_M' columns!"
    
    # 創建一個空的 DataFrame 來存儲結果
    data_draw_DCD = pd.DataFrame(columns=['Group', 'Diff_Scan_M'] + [f'Diff-{col}' for col in ['L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2']])

    # 根據 Group 分組
    grouped = data_draw_CD_name.groupby('Group')

    # 遍歷每個 Group 進行差異計算
    for group_name, group_data in grouped:
        group_data = group_data.reset_index(drop=True)

        # 遍歷每組內的數據，計算當 Scan_M 變化時的差異
        for i in range(1, len(group_data)):
            current_scan = group_data.iloc[i]['Scan_M']
            previous_scan = group_data.iloc[i-1]['Scan_M']

            # 提取 Scan 和 M 的編號
            current_scan_number = current_scan.split('_')[0]
            previous_scan_number = previous_scan.split('_')[0]
            current_M = current_scan.split('_')[1].split('->')[0]
            previous_M = previous_scan.split('_')[1].split('->')[0]

            # 跳過 Scan 和 M 相同的情況，不進行相減
            if current_scan_number == previous_scan_number and current_M == previous_M:
                continue

            # 判斷是否需要顯示兩個 Scan 還是只顯示一個 Scan（M 不同但 Scan 相同）
            if current_scan_number != previous_scan_number:
                diff_scan_M = f"{previous_scan.split('->')[0]}->{current_scan.split('->')[0]}"
            else:
                diff_scan_M = f"{previous_scan.split('_')[0]}_{previous_M}->{current_M}"

            # 確保相關列都是數值型，進行差異計算
            try:
                current_values = pd.to_numeric(group_data.iloc[i][['L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2']], errors='coerce')
                previous_values = pd.to_numeric(group_data.iloc[i-1][['L1', 'L2', 'L3', 'L4', 'L5', 'A1', 'A2']], errors='coerce')

                # 計算差異
                diff_values = current_values - previous_values

                # 創建一行數據，包括 Group, Diff_Scan_M 和差異數據
                diff_row = pd.Series([group_name, diff_scan_M] + diff_values.tolist(), index=data_draw_DCD.columns)

                # 將結果附加到 data_draw_DCD
                data_draw_DCD = pd.concat([data_draw_DCD, diff_row.to_frame().T], ignore_index=True)
                
            except Exception as e:
                print(f"Error calculating differences for Group {group_name}: {e}")

    # 替換 L1~A2 列中的 None 和 NaN 為空字串
    columns_to_replace = ['Diff-L1', 'Diff-L2', 'Diff-L3', 'Diff-L4', 'Diff-L5', 'Diff-A1', 'Diff-A2']
    data_draw_DCD[columns_to_replace] = data_draw_DCD[columns_to_replace].replace({None: '', np.nan: ''})

    # 排序 Diff_Scan_M
    data_draw_DCD = data_draw_DCD.sort_values(by='Diff_Scan_M', key=lambda col: col.map(custom_sort_key))

    # 返回結果
    return data_draw_DCD



def generate_DCD_plot(column, selected_groups):
    global data_draw_DCD

    if data_draw_DCD is None or data_draw_DCD.empty:
        return "Data not available!"

    if column not in data_draw_DCD.columns:
        return "Column not found!"

    grouped = data_draw_DCD.groupby('Group')

    lines = []
    try:
        mean_val = data_draw_DCD[column].mean()
        std_dev = data_draw_DCD[column].std()
        ucl = mean_val + 3 * std_dev
        lcl = mean_val - 3 * std_dev
        
        group_colors = {
            'Left': 'rgba(0, 0, 255, 0.5)',  # 半透明的藍色
            'Left-Center': 'rgba(0, 255, 0, 0.5)',  # 半透明的綠色
            'Center': 'rgba(255, 0, 0, 0.5)',  # 半透明的紅色
            'Right-Center': 'rgba(255, 165, 0, 0.5)',  # 半透明的橙色
            'Right': 'rgba(128, 0, 128, 0.5)',  # 半透明的紫色
            'Left-Outer': 'rgba(0, 255, 255, 0.5)',  # 半透明的青色
            'Left-Inner': 'rgba(255, 0, 255, 0.5)',  # 半透明的洋紅色
            'Right-Inner': 'rgba(255, 255, 0, 0.5)',  # 半透明的黃色
            'Right-Outer': 'rgba(165, 42, 42, 0.5)'  # 半透明的棕色
        }

        for group_name, group_data in grouped:
            if group_name in selected_groups:
                lines.append(go.Scatter(
                    x=group_data['Diff_Scan_M'],
                    y=group_data[column],
                    mode='lines+markers',
                    name=f'{group_name}',
                    hovertemplate=(
                        'Diff_Scan_M: %{x}<br>' +
                        column + ': %{y}<extra></extra>'
                    ),
                    line=dict(color=group_colors.get(group_name, 'rgba(0, 0, 0, 0.5)'))  # 默認為半透明的黑色
                ))

        lines.extend([
            go.Scatter(
                x=data_draw_DCD['Diff_Scan_M'], y=[mean_val] * len(data_draw_DCD),
                mode='lines', name='Mean', line=dict(dash='dash', color='green')
            ),
            go.Scatter(
                x=data_draw_DCD['Diff_Scan_M'], y=[ucl] * len(data_draw_DCD),
                mode='lines', name='UCL', line=dict(dash='dash', color='red')
            ),
            go.Scatter(
                x=data_draw_DCD['Diff_Scan_M'], y=[lcl] * len(data_draw_DCD),
                mode='lines', name='LCL', line=dict(dash='dash', color='red')
            )
        ])

        layout = go.Layout(
            title=f'Line Plot of {column}',
            xaxis=dict(
                title='Diff_Scan_M',
                categoryorder='array',
                categoryarray=data_draw_DCD['Diff_Scan_M'],
                tickangle = -60
            ),
            yaxis=dict(title=column),
            hovermode='closest'
        )

        fig = go.Figure(data=lines, layout=layout)

        return pio.to_html(fig, full_html=False)
    
    except Exception as e:
        return f"Error generating plot: {str(e)}"


if __name__ == '__main__':
    app.run()


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port = 5000)
