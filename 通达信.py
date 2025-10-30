# pip install pandas numpy openpyxl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import struct
import glob

# -------------------------- 1. 核心参数配置（根据需求修改）--------------------------
TARGET_DATE = "20251030"  # 选股日期（格式：YYYYMMDD）
BOX_DAYS = 20  # 箱体周期（近20个交易日）
VOLUME_GROWTH_RATIO = 1.5  # 成交量较昨日增长≥50%
TURNOVER_RATE_THRESHOLD = 10  # 换手率≥10%（单位：%）
MARKET_CAP_THRESHOLD = 100  # 流通市值≤100亿（单位：亿元）
CHIP_CONCENTRATION = 70  # 筹码集中度≥70%（单位：%）
MA_LIST = [5, 10, 20]  # 要突破的均线周期
BREAK_MA_TYPE = "all"  # "all"=全部站上，"any"=至少一条站上

# -------------------------- 2. 通达信路径配置（关键！）--------------------------
# 自动识别常见通达信安装路径（可手动修改为你的通达信根目录）
TDX_PATHS = [
    r"D:\通达信",
    r"D:\东方财富通",
    r"C:\Program Files\通达信",
    r"C:\Program Files (x86)\通达信",
    r"E:\通达信"
]

# 找到存在的通达信路径
TDX_ROOT = None
for path in TDX_PATHS:
    if os.path.exists(os.path.join(path, "T0002", "dsmarket")):
        TDX_ROOT = path
        break

if not TDX_ROOT:
    # 手动指定通达信根目录（若自动识别失败，修改这里）
    TDX_ROOT = r"你的通达信安装目录"  # 例：r"D:\新通达信"

# 通达信数据目录
DAY_DATA_PATH = os.path.join(TDX_ROOT, "T0002", "dsmarket")  # 日线数据（.day文件）
CHIP_DATA_PATH = os.path.join(TDX_ROOT, "T0002", "chip")  # 筹码数据（.chip文件）
STOCK_BASIC_PATH = os.path.join(TDX_ROOT, "T0002", "hq_cache", "stock_basic.csv")  # 股票基础信息

# -------------------------- 3. 核心工具：解析通达信二进制数据 --------------------------
def parse_tdx_day_file(file_path):
    """解析通达信日线文件（.day），返回个股所有日线数据"""
    # .day文件格式：每32字节为一条记录，字段顺序如下
    # 日期(4字节)、开盘价(4字节)、最高价(4字节)、最低价(4字节)、收盘价(4字节)、成交量(4字节)、成交额(4字节)、 reserved(4字节)
    data = []
    with open(file_path, "rb") as f:
        while True:
            record = f.read(32)
            if not record:
                break
            if len(record) != 32:
                continue
            # 解析字段（通达信价格/成交量为整数，需除以100/1000等转换）
            date = struct.unpack("I", record[0:4])[0]  # 日期：YYYYMMDD格式整数
            open_p = struct.unpack("I", record[4:8])[0] / 100  # 开盘价（除以100转为元）
            high = struct.unpack("I", record[8:12])[0] / 100  # 最高价
            low = struct.unpack("I", record[12:16])[0] / 100  # 最低价
            close = struct.unpack("I", record[16:20])[0] / 100  # 收盘价
            vol = struct.unpack("I", record[20:24])[0]  # 成交量（单位：手）
            amount = struct.unpack("I", record[24:28])[0] / 100  # 成交额（除以100转为元）
            data.append([date, open_p, high, low, close, vol, amount])
    
    # 转为DataFrame
    df = pd.DataFrame(data, columns=["trade_date", "open", "high", "low", "close", "vol", "amount"])
    df["trade_date"] = df["trade_date"].astype(str)  # 日期转为字符串（YYYYMMDD）
    return df.sort_values("trade_date").reset_index(drop=True)

def get_stock_basic_info():
    """获取股票基础信息（代码、名称、流通市值）"""
    # 通达信stock_basic.csv格式：代码,名称,流通股本(万股),总股本(万股),流通市值(万元),总市值(万元),行业,地域,上市日期
    if not os.path.exists(STOCK_BASIC_PATH):
        raise FileNotFoundError(f"股票基础信息文件未找到：{STOCK_BASIC_PATH}")
    
    df = pd.read_csv(STOCK_BASIC_PATH, encoding="gbk", header=None)
    df.columns = ["ts_code", "name", "circulating_share", "total_share", "circulating_market_cap", "total_market_cap", "industry", "region", "list_date"]
    
    # 筛选A股（代码以60/00/30开头）
    df = df[df["ts_code"].str.match(r"^60|^00|^30")]
    # 流通市值转为亿元（通达信单位：万元）
    df["circulating_market_cap"] = df["circulating_market_cap"] / 10000
    return df[["ts_code", "name", "circulating_market_cap"]]

def calculate_ma(df, ma_periods):
    """计算均线（MA5/MA10/MA20）"""
    for period in ma_periods:
        df[f"ma{period}"] = df["close"].rolling(window=period, min_periods=period).mean()
    return df

def parse_tdx_chip_data(ts_code, target_date):
    """解析通达信筹码文件（.chip），获取目标日期70%筹码集中度"""
    # 筹码文件命名格式：600000.chip（股票代码.chip）
    chip_file = os.path.join(CHIP_DATA_PATH, f"{ts_code}.chip")
    if not os.path.exists(chip_file):
        return None  # 部分股票无筹码数据
    
    data = []
    with open(chip_file, "rb") as f:
        while True:
            # 筹码文件每条记录48字节，关键字段：日期(4字节)、70%集中度(4字节)
            record = f.read(48)
            if not record:
                break
            if len(record) != 48:
                continue
            date = struct.unpack("I", record[0:4])[0]  # 日期：YYYYMMDD
            concentration_70 = struct.unpack("f", record[24:28])[0]  # 70%筹码集中度（%）
            data.append([date, concentration_70])
    
    chip_df = pd.DataFrame(data, columns=["trade_date", "chip_concentration_70"])
    chip_df["trade_date"] = chip_df["trade_date"].astype(str)
    # 筛选目标日期的筹码集中度
    target_chip = chip_df[chip_df["trade_date"] == target_date]
    return target_chip["chip_concentration_70"].iloc[0] if not target_chip.empty else None

# -------------------------- 4. 选股核心逻辑（与之前一致，适配新数据来源）--------------------------
def check_box_break(df, target_date, box_days):
    """检查箱体突破：近box_days日收盘价最高价为上沿，目标日收盘价突破"""
    # 筛选目标日前box_days个交易日的数据
    df_target = df[df["trade_date"] < target_date].tail(box_days)
    if len(df_target) < box_days:
        return False, 0.0  # 数据不足
    
    box_upper = df_target["close"].max()
    target_close = df[df["trade_date"] == target_date]["close"].iloc[0]
    is_break = target_close > box_upper * 1.005  # 突破幅度≥0.5%
    break_ratio = (target_close / box_upper - 1) * 100 if is_break else 0.0
    return is_break, box_upper, break_ratio

def check_ma_break(stock_df, target_date, ma_list, break_type):
    """检查均线突破：收盘价是否站上指定均线"""
    target_row = stock_df[stock_df["trade_date"] == target_date]
    if target_row.empty:
        return False
    
    target_close = target_row["close"].iloc[0]
    ma_values = [target_row[f"ma{ma}"].iloc[0] for ma in ma_list]
    
    if break_type == "all":
        return all(target_close > ma for ma in ma_values)
    elif break_type == "any":
        return any(target_close > ma for ma in ma_values)
    return False

def get_prev_trade_date(df, target_date):
    """获取目标日期的前一个交易日（排除节假日）"""
    trade_dates = df["trade_date"].tolist()
    if target_date not in trade_dates:
        return None
    target_idx = trade_dates.index(target_date)
    return trade_dates[target_idx - 1] if target_idx > 0 else None

# -------------------------- 5. 主选股流程 --------------------------
def stock_selection_tdx_auto():
    print(f"开始执行通达信原生数据选股：日期={TARGET_DATE}")
    print(f"选股条件：")
    print(f"1. 当日成交量较昨日增长≥50%")
    print(f"2. 当日换手率≥{TURNOVER_RATE_THRESHOLD}%")
    print(f"3. 流通市值≤{MARKET_CAP_THRESHOLD}亿")
    print(f"4. 突破{BOX_DAYS}日箱体")
    print(f"5. 筹码集中度≥{CHIP_CONCENTRATION}%")
    print(f"6. 均线突破（{BREAK_MA_TYPE}站上{MA_LIST}日均线）")
    
    # 步骤1：获取股票基础信息（筛选市值≤100亿）
    try:
        basic_df = get_stock_basic_info()
        basic_df = basic_df[basic_df["circulating_market_cap"] <= MARKET_CAP_THRESHOLD]
        ts_codes = basic_df["ts_code"].tolist()
        print(f"\n步骤1：流通市值≤{MARKET_CAP_THRESHOLD}亿的股票共 {len(ts_codes)} 只")
        if len(ts_codes) == 0:
            print("无符合市值条件的股票，退出")
            return
    except Exception as e:
        print(f"获取股票基础信息失败：{e}")
        return
    
    # 步骤2：遍历每只股票，解析数据并筛选
    final_selected = []
    total = len(ts_codes)
    for i, ts_code in enumerate(ts_codes):
        if i % 100 == 0:
            print(f"正在处理第 {i}/{total} 只股票：{ts_code}")
        
        # 步骤2.1：读取个股日线数据
        day_file = glob.glob(os.path.join(DAY_DATA_PATH, f"{ts_code}*.day"))  # 适配不同市场后缀（如.SZ/.SH）
        if not day_file:
            continue
        day_df = parse_tdx_day_file(day_file[0])
        
        # 步骤2.2：检查目标日期是否存在
        if TARGET_DATE not in day_df["trade_date"].values:
            continue
        
        # 步骤2.3：获取前一个交易日数据（用于计算成交量增长）
        prev_date = get_prev_trade_date(day_df, TARGET_DATE)
        if not prev_date or prev_date not in day_df["trade_date"].values:
            continue
        
        # 步骤2.4：计算均线
        day_df = calculate_ma(day_df, MA_LIST)
        # 检查均线数据是否完整（目标日需有所有均线值）
        target_row = day_df[day_df["trade_date"] == TARGET_DATE]
        if any(pd.isna(target_row[f"ma{ma}"].iloc[0]) for ma in MA_LIST):
            continue
        
        # 步骤2.5：提取关键数据
        target_close = target_row["close"].iloc[0]
        target_vol = target_row["vol"].iloc[0]
        prev_vol = day_df[day_df["trade_date"] == prev_date]["vol"].iloc[0]
        target_amount = target_row["amount"].iloc[0]
        circulating_market_cap = basic_df[basic_df["ts_code"] == ts_code]["circulating_market_cap"].iloc[0]
        stock_name = basic_df[basic_df["ts_code"] == ts_code]["name"].iloc[0]
        
        # 计算换手率（换手率=成交量×100 / 流通股本（万股））
        circulating_share = basic_df[basic_df["ts_code"] == ts_code]["circulating_share"].iloc[0]  # 流通股本（万股）
        turnover_rate = (target_vol * 100) / circulating_share  # 成交量单位：手→股（×100），除以流通股本（万股）→%
        
        # 步骤2.6：筛选条件
        # 条件1：成交量增长≥50%
        volume_growth = target_vol / prev_vol
        if volume_growth < VOLUME_GROWTH_RATIO:
            continue
        
        # 条件2：换手率≥10%
        if turnover_rate < TURNOVER_RATE_THRESHOLD:
            continue
        
        # 条件3：筹码集中度≥70%
        chip_concentration = parse_tdx_chip_data(ts_code, TARGET_DATE)
        if not chip_concentration or chip_concentration < CHIP_CONCENTRATION:
            continue
        
        # 条件4：箱体突破
        is_box_break, box_upper, break_ratio = check_box_break(day_df, TARGET_DATE, BOX_DAYS)
        if not is_box_break:
            continue
        
        # 条件5：均线突破
        if not check_ma_break(day_df, TARGET_DATE, MA_LIST, BREAK_MA_TYPE):
            continue
        
        # 收集结果
        final_selected.append({
            "股票代码": ts_code,
            "股票名称": stock_name,
            "流通市值（亿元）": round(circulating_market_cap, 2),
            "当日收盘价（元）": round(target_close, 2),
            "5日均线（元）": round(target_row["ma5"].iloc[0], 2) if 5 in MA_LIST else "-",
            "10日均线（元）": round(target_row["ma10"].iloc[0], 2) if 10 in MA_LIST else "-",
            "20日均线（元）": round(target_row["ma20"].iloc[0], 2) if 20 in MA_LIST else "-",
            "成交量增长（%）": round((volume_growth - 1) * 100, 2),
            "当日换手率（%）": round(turnover_rate, 2),
            "筹码集中度（%）": round(chip_concentration, 2),
            "箱体上沿（元）": round(box_upper, 2),
            "箱体突破幅度（%）": round(break_ratio, 2)
        })
    
    # 输出结果
    final_df = pd.DataFrame(final_selected)
    print(f"\n==================== 最终选股结果（共 {len(final_df)} 只）====================")
    if len(final_df) > 0:
        print(final_df.to_string(index=False))
        # 保存结果到通达信根目录下的“选股结果”文件夹
        save_dir = os.path.join(TDX_ROOT, "选股结果")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"通达信选股结果_{TARGET_DATE}.xlsx")
        final_df.to_excel(save_path, index=False)
        print(f"\n结果已保存至：{save_path}")
    else:
        print("无同时满足所有条件的股票")

# -------------------------- 6. 执行选股 --------------------------
if __name__ == "__main__":
    try:
        stock_selection_tdx_auto()
    except Exception as e:
        print(f"选股过程出错：{e}")
        print("请检查：1. 通达信路径是否正确；2. 目标日期是否有交易数据；3. 通达信是否已下载完整日线/筹码数据")
