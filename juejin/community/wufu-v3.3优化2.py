# 克隆自聚宽文章：https://www.joinquant.com/post/1399
# 标题：【量化课堂】多因子策略入门
# 作者：JoinQuant量化课堂

# 克隆自聚宽文章：https://www.joinquant.com/post/69132
# 标题：给【五福闹新春】v3.3策略做了一次代码提速，回测快了70%
# 作者：忆似水

# 克隆自聚宽文章：https://www.joinquant.com/post/68831
# 标题：【五福闹新春】v3.3-潮生潮落帆来去，云卷云舒山有无
# 作者：烟花三月ETF

# 克隆自聚宽文章：https://www.joinquant.com/post/68831
# 标题：【五福闹新春】v3.3-潮生潮落帆来去，云卷云舒山有无
# 作者：烟花三月ETF
# 修改说明：
# 1. RSI过滤增强：检查过去5个交易日，若任一天RSI超过阈值，则要求当前价格 > 3日均线，否则剔除。
# 2. 新增溢价率过滤：可设置阈值，剔除溢价率过高的ETF/LOF（基于前一日净值计算）。

import numpy as np
import math
import pandas as pd
from jqdata import *
from datetime import datetime, date


# ==================== 策略初始化 ====================
def initialize(context):
    """初始化策略（设置参数、全局变量、定时任务）"""
    set_option("avoid_future_data", True)       # 避免未来函数
    set_option("use_real_price", True)          # 使用真实价格
    
    set_slippage(PriceRelatedSlippage(0.0001), type="fund")  # 设置滑点
    
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0001,
                              close_commission=0.0001, close_today_commission=0.0001,
                              min_commission=5), type="fund")  # 设置交易费用

    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'info')
    log.info("【五福闹新春】v3.3（增强版：RSI+溢价率过滤）启动！")

    set_benchmark("510300.XSHG")                # 设置基准

    # ==================== 固定ETF池 ====================
    g.fixed_etf_pool = [
#大宗商品ETF：
        '518880.XSHG',  # (黄金ETF) [ETF]-成交额：54.60亿元-上市日期：2013-07-29
        '161226.XSHE',  # (国投白银LOF) [LOF]-成交额：21.54亿元-上市日期：2015-08-17
        '159980.XSHE',  # (有色ETF大成) [ETF]-成交额：23.57亿元-上市日期：2019-12-24
        '501018.XSHG',  # (南方原油ETF) [LOF]-成交额：1.34亿元-上市日期：2016-06-28
        '159985.XSHE',  # (豆粕ETF) [ETF]-成交额：0.67亿元
#海外ETF：       
        '513100.XSHG',  # (纳指ETF) [ETF]-成交额：4.24亿元-上市日期：2013-05-15
        '159509.XSHE',  # (纳指科技ETF景顺) [ETF]-成交额：5.65亿元-上市日期：2023-08-08
        '513290.XSHG',  # (纳指生物) [ETF]-成交额：1.28亿元-上市日期：2022-08-29        
        '513500.XSHG',  # (标普500) [ETF]-成交额：2.22亿元-上市日期：2014-01-15
        '159518.XSHE',  # (标普油气ETF嘉实) [ETF]-成交额：5.35亿元-上市日期：2023-11-15
        '159502.XSHE',  # (标普生物科技ETF嘉实) [ETF]-成交额：4.00亿元-上市日期：2024-01-10        
        '159529.XSHE',  # (标普消费ETF) [ETF]-成交额：2.25亿元-上市日期：2024-02-02
        '513400.XSHG',  # (道琼斯) [ETF]-成交额：1.09亿元-上市日期：2024-02-02
        '520830.XSHG',  # (沙特ETF) [ETF]-成交额：1.16亿元-上市日期：2024-07-16
        '513520.XSHG',  # (日经ETF) [ETF]-成交额：1.11亿元-上市日期：2019-06-25
        '513030.XSHG',  # (德国ETF) [ETF]-成交额：0.77亿元
#港股ETF：
        '513090.XSHG',  # (香港证券) [ETF]-成交额：68.32亿元-上市日期：2020-03-26
        '513180.XSHG',  # (恒指科技) [ETF]-成交额：61.72亿元-上市日期：2021-05-25
        '513120.XSHG',  # (HK创新药) [ETF]-成交额：48.95亿元-上市日期：2022-07-12
        '513330.XSHG',  # (恒生互联) [ETF]-成交额：37.01亿元-上市日期：2021-02-08
        '513750.XSHG',  # (港股非银) [ETF]-成交额：23.06亿元-上市日期：2023-11-27
        '159892.XSHE',  # (恒生医药ETF) [ETF]-成交额：12.25亿元-上市日期：2021-10-19
        '159605.XSHE',  # (中概互联ETF) [ETF]-成交额：5.14亿元-上市日期：2021-12-02
        '513190.XSHG',  # (H股金融) [ETF]-成交额：5.07亿元-上市日期：2023-10-11
        '510900.XSHG',  # (恒生中国) [ETF]-成交额：3.73亿元-上市日期：2012-10-22
        '513630.XSHG',  # (香港红利) [ETF]-成交额：3.69亿元-上市日期：2023-12-08
        '513920.XSHG',  # (港股通央企红利) [ETF]-成交额：3.11亿元-上市日期：2024-01-05
        '159323.XSHE',  # (港股通汽车ETF) [ETF]-成交额：2.02亿元-上市日期：2025-01-08
        '513970.XSHG',  # (恒生消费) [ETF]-成交额：1.25亿元-上市日期：2023-04-21
#指数ETF：        
        '510500.XSHG',  # (中证500ETF) [ETF]-成交额：263.30亿元-上市日期：2013-03-15
        '512100.XSHG',  # (中证1000ETF) [ETF]-成交额：32.30亿元-上市日期：2016-11-04
        '563300.XSHG',  # (中证2000) [ETF]-成交额：3.34亿元-上市日期：2023-09-14        
        '510300.XSHG',  # (沪深300ETF) [ETF]-成交额：253.91亿元-上市日期：2012-05-28
        '512050.XSHG',  # (A500E) [ETF]-成交额：151.68亿元-上市日期：2024-11-15        
        '510760.XSHG',  # (上证ETF) [ETF]-成交额：1.10亿元-上市日期：2020-09-09        
        '159915.XSHE',  # (创业板ETF易方达) [ETF]-成交额：129.05亿元-上市日期：2011-12-09
        '159949.XSHE',  # (创业板50ETF) [ETF]-成交额：15.23亿元-上市日期：2016-07-22
        '159967.XSHE',  # (创业板成长ETF) [ETF]-成交额：3.27亿元-上市日期：2019-07-15        
        '588080.XSHG',  # (科创板50) [ETF]-成交额：123.46亿元-上市日期：2020-11-16
        '588220.XSHG',  # (科创100) [ETF]-成交额：4.99亿元-上市日期：2023-09-15
        '511380.XSHG',  # (可转债ETF) [ETF]-成交额：165.76亿元-上市日期：2020-04-07
#行业ETF：
        '513310.XSHG',  # (中韩芯片) [ETF]-成交额：38.68亿元-上市日期：2022-12-22
        '588200.XSHG',  # (科创芯片) [ETF]-成交额：37.94亿元-上市日期：2022-10-26
        '159852.XSHE',  # (软件ETF) [ETF]-成交额：36.26亿元-上市日期：2021-02-09
        '512880.XSHG',  # (证券ETF) [ETF]-成交额：34.01亿元-上市日期：2016-08-08
        '159206.XSHE',  # (卫星ETF) [ETF]-成交额：32.60亿元-上市日期：2025-03-14
        '512400.XSHG',  # (有色金属ETF) [ETF]-成交额：31.27亿元-上市日期：2017-09-01
        '512980.XSHG',  # (传媒ETF) [ETF]-成交额：30.96亿元-上市日期：2018-01-19
        '159516.XSHE',  # (半导体设备ETF) [ETF]-成交额：28.21亿元-上市日期：2023-07-27
        '512480.XSHG',  # (半导体) [ETF]-成交额：16.29亿元-上市日期：2019-06-12
        '515880.XSHG',  # (通信ETF) [ETF]-成交额：13.46亿元-上市日期：2019-09-06
        '562500.XSHG',  # (机器人) [ETF]-成交额：12.92亿元-上市日期：2021-12-29
        '159218.XSHE',  # (卫星产业ETF) [ETF]-成交额：12.74亿元-上市日期：2025-05-22
        '159869.XSHE',  # (游戏ETF) [ETF]-成交额：12.42亿元-上市日期：2021-03-05
        '159870.XSHE',  # (化工ETF) [ETF]-成交额：12.30亿元-上市日期：2021-03-03
        '159326.XSHE',  # (电网设备ETF) [ETF]-成交额：12.02亿元-上市日期：2024-09-09
        '159851.XSHE',  # (金融科技ETF) [ETF]-成交额：11.79亿元-上市日期：2021-03-19
        '560860.XSHG',  # (工业有色) [ETF]-成交额：11.71亿元-上市日期：2023-03-13
        '159363.XSHE',  # (创业板人工智能ETF华宝) [ETF]-成交额：10.63亿元-上市日期：2024-12-16
        '588170.XSHG',  # (科创半导) [ETF]-成交额：10.28亿元-上市日期：2025-04-08
        '159755.XSHE',  # (电池ETF) [ETF]-成交额：10.02亿元-上市日期：2021-06-24
        '512170.XSHG',  # (医疗ETF) [ETF]-成交额：9.54亿元-上市日期：2019-06-17
        '512800.XSHG',  # (银行ETF) [ETF]-成交额：9.48亿元-上市日期：2017-08-03
        '159819.XSHE',  # (人工智能ETF易方达) [ETF]-成交额：9.40亿元-上市日期：2020-09-23
        '512710.XSHG',  # (军工龙头) [ETF]-成交额：9.39亿元-上市日期：2019-08-26
        '159638.XSHE',  # (高端装备ETF嘉实) [ETF]-成交额：8.92亿元-上市日期：2022-08-12
        '517520.XSHG',  # (黄金股) [ETF]-成交额：8.73亿元-上市日期：2023-11-01
        '515980.XSHG',  # (人工智能) [ETF]-成交额：8.73亿元-上市日期：2020-02-10
        '159995.XSHE',  # (芯片ETF) [ETF]-成交额：8.45亿元-上市日期：2020-02-10
        '159227.XSHE',  # (航空航天ETF) [ETF]-成交额：8.42亿元-上市日期：2025-05-16
        '512660.XSHG',  # (军工ETF) [ETF]-成交额：7.78亿元-上市日期：2016-08-08
        '512690.XSHG',  # (酒ETF) [ETF]-成交额：6.74亿元-上市日期：2019-05-06
        '516150.XSHG',  # (稀土基金) [ETF]-成交额：6.41亿元-上市日期：2021-03-17
        '512890.XSHG',  # (红利低波) [ETF]-成交额：6.03亿元-上市日期：2019-01-18
        '588790.XSHG',  # (科创智能) [ETF]-成交额：5.92亿元-上市日期：2025-01-09
        '159992.XSHE',  # (创新药ETF) [ETF]-成交额：5.63亿元-上市日期：2020-04-10
        '512070.XSHG',  # (证券保险) [ETF]-成交额：5.50亿元-上市日期：2014-07-18
        '562800.XSHG',  # (稀有金属) [ETF]-成交额：5.49亿元-上市日期：2021-09-27
        '512010.XSHG',  # (医药ETF) [ETF]-成交额：5.22亿元-上市日期：2013-10-28
        '515790.XSHG',  # (光伏ETF) [ETF]-成交额：4.95亿元-上市日期：2020-12-18
        '510880.XSHG',  # (红利ETF) [ETF]-成交额：4.90亿元-上市日期：2007-01-18
        '159928.XSHE',  # (消费ETF) [ETF]-成交额：4.71亿元-上市日期：2013-09-16
        '159883.XSHE',  # (医疗器械ETF) [ETF]-成交额：4.44亿元-上市日期：2021-04-30
        '159998.XSHE',  # (计算机ETF) [ETF]-成交额：3.93亿元-上市日期：2020-04-13
        '515220.XSHG',  # (煤炭ETF) [ETF]-成交额：3.92亿元-上市日期：2020-03-02
        '561980.XSHG',  # (芯片设备) [ETF]-成交额：3.89亿元-上市日期：2023-09-01
        '515400.XSHG',  # (大数据) [ETF]-成交额：3.54亿元-上市日期：2021-01-20
        '515120.XSHG',  # (创新药) [ETF]-成交额：3.54亿元-上市日期：2021-01-04
        '159566.XSHE',  # (储能电池ETF易方达) [ETF]-成交额：3.05亿元-上市日期：2024-02-08
        '515050.XSHG',  # (5GETF) [ETF]-成交额：3.04亿元-上市日期：2019-10-16
        '516510.XSHG',  # (云计算ETF) [ETF]-成交额：2.95亿元-上市日期：2021-04-07
        '159256.XSHE',  # (创业板软件ETF华夏) [ETF]-成交额：2.89亿元-上市日期：2025-08-04
        '159766.XSHE',  # (旅游ETF) [ETF]-成交额：2.57亿元-上市日期：2021-07-23
        '512200.XSHG',  # (地产ETF) [ETF]-成交额：2.53亿元-上市日期：2017-09-25
        '513350.XSHG',  # (油气ETF) [ETF]-成交额：2.48亿元-上市日期：2023-11-28
        '159583.XSHE',  # (通信设备ETF) [ETF]-成交额：2.47亿元-上市日期：2024-07-08
        '159732.XSHE',  # (消费电子ETF) [ETF]-成交额：2.39亿元-上市日期：2021-08-23
        '516160.XSHG',  # (新能源) [ETF]-成交额：2.26亿元-上市日期：2021-02-04
        '516520.XSHG',  # (智能驾驶) [ETF]-成交额：2.22亿元-上市日期：2021-03-01
        '562590.XSHG',  # (半导材料) [ETF]-成交额：1.94亿元-上市日期：2023-10-18
        '515030.XSHG',  # (新汽车) [ETF]-成交额：1.93亿元-上市日期：2020-03-04
        '512670.XSHG',  # (国防ETF) [ETF]-成交额：1.84亿元-上市日期：2019-08-01
        '561330.XSHG',  # (矿业ETF) [ETF]-成交额：1.81亿元-上市日期：2022-11-01
        '516190.XSHG',  # (文娱ETF) [ETF]-成交额：1.67亿元-上市日期：2021-09-17
        '159840.XSHE',  # (锂电池ETF工银) [ETF]-成交额：1.61亿元-上市日期：2021-08-20
        '159611.XSHE',  # (电力ETF) [ETF]-成交额：1.52亿元-上市日期：2022-01-07
        '159981.XSHE',  # (能源化工ETF) [ETF]-成交额：1.48亿元-上市日期：2020-01-17
        '159865.XSHE',  # (养殖ETF) [ETF]-成交额：1.40亿元-上市日期：2021-03-08
        '561360.XSHG',  # (石油ETF) [ETF]-成交额：1.36亿元-上市日期：2023-10-31
        '159667.XSHE',  # (工业母机ETF) [ETF]-成交额：1.32亿元-上市日期：2022-10-26
        '515170.XSHG',  # (食品饮料ETF) [ETF]-成交额：1.30亿元-上市日期：2021-01-13
        '513360.XSHG',  # (教育ETF) [ETF]-成交额：1.09亿元-上市日期：2021-06-17
        '159825.XSHE',  # (农业ETF) [ETF]-成交额：1.05亿元-上市日期：2020-12-29
        '515210.XSHG',  # (钢铁ETF) [ETF]-成交额：1.03亿元-上市日期：2020-03-02
    ]

    g.filtered_fixed_pool = []           # 过滤后的固定ETF池
    g.dynamic_etf_pool = []              # 动态ETF池（初始为空）
    g.merged_etf_pool = []               # 合并后的ETF池
    g.ranked_etfs_result = []            # 动量计算结果的ETF列表
    g.last_refresh_date = None           # 上次刷新日期
    g.positions = {}                     # 记录目标持仓  
    
    # ==================== 策略核心参数 ====================
    g.holdings_num = 1                  # 持仓数量
    g.defensive_etf = "511880.XSHG"     # 防御型ETF（市场低迷时持有）
    g.safe_haven_etf = '511660.XSHG'    # 冷却期避险ETF
    g.min_money = 5000                  # 最小交易金额（元）

    # 动量计算参数
    g.lookback_days = 25                # 动量计算回看天数
    g.min_score_threshold = 0           # 动量得分下限
    g.max_score_threshold = 5           # 动量得分上限
    g.score_threshold_ratio = 0.9       # 减少调仓控制得分比例

    # 过滤开关及参数
    g.use_short_momentum_filter = False  # 短期动量过滤开关
    g.short_lookback_days = 10           # 短期动量回看天数
    g.short_momentum_threshold = 0.0     # 短期动量阈值

    g.enable_r2_filter = True            # R²过滤开关
    g.r2_threshold = 0.4                 # R²阈值

    g.enable_annualized_return_filter = False  # 年化收益过滤开关
    g.min_annualized_return = 1.0        # 年化收益阈值

    g.enable_ma_filter = False           # 均线过滤开关
    g.ma_filter_days = 20                # 均线周期

    g.enable_volume_check = True         # 成交量过滤开关
    g.volume_lookback = 5                # 成交量回看天数
    g.volume_threshold = 1.0             # 成交量比阈值

    g.enable_loss_filter = True          # 短期风控过滤开关
    g.loss = 0.97                        # 单日最大允许跌幅（3%）

    # ========== 增强的RSI过滤参数 ==========
    g.use_rsi_filter = False              # RSI过滤开关（开启后使用增强逻辑）
    g.rsi_period = 6                     # RSI周期
    g.rsi_lookback_days = 5              # RSI回看天数（检查过去N天是否超买）
    g.rsi_threshold = 98                 # RSI超买阈值
    g.rsi_ma_period = 3                  # 用于过滤的均线周期（当前价>此均线则保留）

    # ========== 溢价率过滤参数 ==========
    g.enable_premium_filter = True       # 溢价率过滤开关
    g.premium_threshold = 15.0           # 溢价率阈值（百分比），超过该值则剔除。可根据需要调整（如5%、10%等）
    g.premium_penalty = False            # 是否对高溢价进行惩罚（扣分）而不是直接剔除（本策略采用直接剔除）

    # 止损参数
    g.use_fixed_stop_loss = True         # 固定比例止损开关
    g.fixedStopLossThreshold = 0.95      # 固定止损比例（5%）
    g.use_pct_stop_loss = False          # 当日跌幅止损开关
    g.pct_stop_loss_threshold = 0.95     # 当日跌幅止损比例

    # 冷却期参数
    g.sell_cooldown_enabled = False      # 卖出冷却期开关
    g.sell_cooldown_days = 3             # 冷却期天数
    g.cooldown_end_date = None           # 冷却期结束日期
    
    # ==================== 流动性阈值设置 ====================
    # 动态阈值模式：每天08:50准时更新，初始化时不计算（由定时任务保证）
    g.avg_etf_money_threshold = None
    log.info("【流动性阈值模式】使用动态阈值（每天08:50自动更新）")
    
    # ==================== 定时任务 ====================
    run_weekly(calculate_global_etf_threshold, 1,time='08:50')   # 每天8:50更新阈值
    run_weekly(update_sector_pool, 1,time='09:01')               # 每天9:01更新动态池
    run_weekly(filter_fixed_pool_by_volume, 1,time='09:02')      # 每天9:02过滤固定池
    run_weekly(daily_merge_etf_pools, 1,time='09:03')            # 每天9:03合并池
    #run_daily(calculate_global_etf_threshold, time='08:50')   # 每天8:50更新阈值
    #run_daily(update_sector_pool, time='09:01')               # 每天9:01更新动态池
    #run_daily(filter_fixed_pool_by_volume, time='09:02')      # 每天9:02过滤固定池
    #run_daily(daily_merge_etf_pools, time='09:03')            # 每天9:03合并池    
    run_daily(check_positions, time='09:04')                  # 每天9:04盘前检查持仓
    run_daily(calculate_and_log_ranked_etfs, time='13:09:59') # 计算动量得分
    run_daily(execute_sell_trades, time='13:10:00')           # 执行卖出
    run_daily(execute_buy_trades, time='13:11:00')            # 执行买入

    # 分钟级止损任务（每1分钟检查一次）
    for hour in range(9, 15):
        for minute in range(0, 60):
            current_time = "%02d:%02d" % (hour, minute)
            if current_time == '09:30' or current_time == '10:30' or current_time == '14:00' or current_time == '14:57':
            #if ('09:25' < current_time < '11:30') or ('13:00' < current_time < '14:57'):
                run_daily(minute_level_stop_loss, time=current_time)      # 固定比例止损
                #run_daily(minute_level_pct_stop_loss, time=current_time)  # 当日跌幅止损
    
    # 日志输出
    log.info(f"""策略参数初始化完成:
=== 过滤条件 ===
- 动量得分过滤: {'启用' if (g.min_score_threshold > -1e9 or g.max_score_threshold < 1e9) else '禁用'} (阈值范围: [{g.min_score_threshold}, {g.max_score_threshold}])
- 短期动量过滤: {'启用' if g.use_short_momentum_filter else '禁用'} (周期: {g.short_lookback_days}天, 阈值 ≥ {g.short_momentum_threshold:.2f})
- R²过滤: {'启用' if g.enable_r2_filter else '禁用'} (阈值 > {g.r2_threshold:.1f})
- 年化收益率过滤: {'启用' if g.enable_annualized_return_filter else '禁用'} (阈值 ≥ {g.min_annualized_return:.0%})
- 均线过滤: {'启用' if g.enable_ma_filter else '禁用'} ({g.ma_filter_days}日均线)
- 成交量过滤: {'启用' if g.enable_volume_check else '禁用'} (近{g.volume_lookback}日均量比 < {g.volume_threshold:.1f})
- 短期风控过滤: {'启用' if g.enable_loss_filter else '禁用'} (近3日单日跌幅 < {1 - g.loss:.0%})
- RSI过滤(增强): {'启用' if g.use_rsi_filter else '禁用'} (周期: {g.rsi_period}, 回看{g.rsi_lookback_days}日, 触发阈值 > {g.rsi_threshold}, 需价格>{g.rsi_ma_period}日均线)
- 溢价率过滤: {'启用' if g.enable_premium_filter else '禁用'} (阈值: {g.premium_threshold}%)
- 流动性门槛: 近3日日均成交额 ≥ 动态阈值（每天08:50自动更新）
- 减少调仓控制得分比例: {g.score_threshold_ratio}（第{g.holdings_num}名得分 × 此比例）

=== 止损机制 ===
- 分钟级固定比例止损: {'启用' if g.use_fixed_stop_loss else '禁用'} (持仓成本价 × {g.fixedStopLossThreshold:.2%})
- 分钟级当日跌幅止损: {'启用' if g.use_pct_stop_loss else '禁用'} (昨日收盘价 × {g.pct_stop_loss_threshold:.2%})

=== 其他配置 ===
- 固定ETF池: {len(g.fixed_etf_pool)} 只
- 持仓数量: {g.holdings_num}只
- 防御ETF: {g.defensive_etf}
- 冷却期避险ETF: {g.safe_haven_etf}
- 冷却期: {'启用' if g.sell_cooldown_enabled else '禁用'}
""")


# ==================== 溢价率计算函数 ====================
def get_premium_rate(context, etf_code):
    """
    计算ETF/LOF的溢价率
    使用前一日单位净值计算
    返回: (溢价率百分比, 前一日净值, 是否成功)
    """
    try:
        # 获取前一日收盘价
        etf_price_df = get_price(etf_code, start_date=context.previous_date, end_date=context.previous_date, fields=['close'])
        if len(etf_price_df) == 0:
            return 0, 0, False
        
        etf_price = etf_price_df['close'].iloc[-1]
        
        # 获取前一日单位净值（使用聚宽的get_extras接口）
        nav_df = get_extras('unit_net_value', etf_code, start_date=context.previous_date, end_date=context.previous_date)
        if nav_df is None or len(nav_df) == 0:
            return 0, 0, False
        
        nav = nav_df.iloc[-1].values[0]
        
        # 计算溢价率
        if nav is not None and nav != 0 and not np.isnan(nav):
            premium_rate = (etf_price - nav) / nav * 100
            return premium_rate, nav, True
        else:
            return 0, 0, False
            
    except Exception as e:
        log.debug(f"{etf_code} 计算溢价率失败: {e}")
        return 0, 0, False


# ==================== 增强的RSI过滤函数 ====================
def check_rsi_filter_enhanced(price_series, current_price, context):
    """
    增强的RSI过滤：检查过去 g.rsi_lookback_days 个交易日
    如果其中任一天RSI超过 g.rsi_threshold，则需要当前价格 > g.rsi_ma_period 日均线
    否则通过（如果没有超过阈值，则通过）
    返回: (passed, max_recent_rsi, 说明)
    """
    if not g.use_rsi_filter:
        return True, 0, "RSI过滤未启用"
    
    if len(price_series) < g.rsi_period + g.rsi_lookback_days:
        return True, 0, f"数据不足{len(price_series)}"
    
    # 计算RSI序列
    rsi_values = calculate_rsi(price_series, g.rsi_period)
    if len(rsi_values) < g.rsi_lookback_days:
        return True, 0, "RSI计算数据不足"
    
    # 获取最近 N 天的RSI值
    recent_rsi = rsi_values[-g.rsi_lookback_days:]
    max_recent_rsi = np.max(recent_rsi)
    
    # 检查是否有任何一天超过阈值
    if np.any(np.array(recent_rsi) > g.rsi_threshold):
        # 超过阈值，需要检查价格与均线关系
        if len(price_series) >= g.rsi_ma_period:
            ma = np.mean(price_series[-g.rsi_ma_period:])
            if current_price > ma:
                return True, max_recent_rsi, f"RSI超买(最高{max_recent_rsi:.1f})但价格>{g.rsi_ma_period}日均线{ma:.2f}"
            else:
                return False, max_recent_rsi, f"RSI超买(最高{max_recent_rsi:.1f})且价格{current_price:.2f} <= {g.rsi_ma_period}日均线{ma:.2f}"
        else:
            return False, max_recent_rsi, f"RSI超买但均线数据不足"
    else:
        return True, max_recent_rsi, f"RSI未超买(最高{max_recent_rsi:.1f})"


# ==================== 过滤条件应用 ====================
def apply_filters(metrics_list):
    """根据开关应用所有过滤条件（包括增强RSI和溢价率）"""
    steps = [
        ('动量得分', lambda m: m['passed_momentum'], True),
        ('短期动量', lambda m: m['passed_short_mom'], g.use_short_momentum_filter),
        ('R²', lambda m: m['passed_r2'], g.enable_r2_filter),
        ('年化收益率', lambda m: m['passed_annual_ret'], g.enable_annualized_return_filter),
        ('均线', lambda m: m['passed_ma'], g.enable_ma_filter),
        ('成交量', lambda m: m['passed_volume'], g.enable_volume_check),
        ('短期风控', lambda m: m['passed_loss'], g.enable_loss_filter),
        ('RSI(增强)', lambda m: m['passed_rsi'], g.use_rsi_filter),          # 增强RSI过滤
        ('溢价率', lambda m: m['passed_premium'], g.enable_premium_filter),   # 溢价率过滤
    ]
    filtered = metrics_list[:]
    for _, condition, is_enabled in steps:
        if is_enabled:
            filtered = [m for m in filtered if condition(m)]
    return filtered


# ==================== 持仓检查 ====================
def check_positions(context):
    """盘前持仓检查"""
    current_data = get_current_data()
    for security in context.portfolio.positions:
        position = context.portfolio.positions[security]
        if position.total_amount > 0:
            security_name = get_security_name(security)
            log.info(f"📊 持仓检查: {security} {security_name}, 数量: {position.total_amount}, "
                     f"成本: {position.avg_cost:.3f}, 当前价: {position.price:.3f}")
            if current_data[security].paused:
                log.info(f"⚠️ {security} {security_name} 今日停牌")


# ==================== 流动性阈值计算 ====================
def calculate_global_etf_threshold(context):
    """计算全市场ETF流动性阈值（每天08:50准时更新）"""
    log.info("★" * 80)
    log.info("【全局阈值更新】开始计算全市场ETF流动性门槛")

    try:
        # 获取所有ETF
        all_funds = get_all_securities(['fund'], date=context.current_dt).index.tolist()
        etf_list = []
        for code in all_funds:
            try:
                info = get_security_info(code)
                if info and info.subtype == 'etf':
                    etf_list.append(code)
            except Exception:
                continue

        if not etf_list:
            log.warning("未找到任何场内ETF，使用保守阈值1000万")
            g.avg_etf_money_threshold = 10000000
            return

        log.info(f"全市场ETF总数: {len(etf_list)} 只")

        # 获取最近3个交易日
        current_date = context.current_dt.date()
        trade_days = get_trade_days(end_date=current_date - pd.Timedelta(days=1), count=3)

        if len(trade_days) < 3:
            log.warning("无法获取3个完整交易日，使用保守阈值1000万")
            g.avg_etf_money_threshold = 10000000
            return

        daily_totals = []
        valid_days = 0
        for day in trade_days:
            try:
                df = get_price(security=etf_list, start_date=day, end_date=day, frequency='daily',
                               fields=['money'], panel=False, skip_paused=True)
                if df is not None and not df.empty:
                    daily_total = df['money'].sum()
                    daily_totals.append(daily_total)
                    log.info(f"{day} 全市场ETF总成交额: {daily_total/1e8:.2f}亿元 "
                             f"({df['money'].count()}只ETF有成交)")
                    valid_days += 1
            except Exception as e:
                log.warning(f"计算 {day} 成交额失败: {e}")

        if valid_days < 3:
            log.warning(f"仅有 {valid_days} 个有效交易日，使用保守阈值1000万")
            g.avg_etf_money_threshold = 10000000
            return

        avg_total_money = sum(daily_totals) / len(daily_totals)
        threshold = avg_total_money / 20000
        g.avg_etf_money_threshold = threshold

        log.info(f"【全局阈值更新完成】近3日全市场ETF日均总成交额 = {avg_total_money/1e8:.2f}亿元，"
                 f"阈值 = {threshold/1e4:.0f}万元 ({threshold:,.0f}元)")

    except Exception as e:
        log.warning(f"计算全局阈值异常: {e}，使用保守阈值1000万")
        g.avg_etf_money_threshold = 10000000


# ==================== 固定池流动性过滤 ====================
def filter_fixed_pool_by_volume(context):
    """每日对固定ETF池进行流动性过滤"""
    log.info("=" * 70)
    log.info("【固定池过滤】开始执行")

    # 确保阈值已初始化
    if g.avg_etf_money_threshold is None:
        log.info("【固定池过滤】阈值未初始化，立即计算")
        calculate_global_etf_threshold(context)

    if not g.fixed_etf_pool:
        log.info("【固定池过滤】固定池为空，跳过过滤")
        return

    dynamic_threshold = g.avg_etf_money_threshold
    log.info(f"【固定池过滤】使用流动性门槛 = 日均{dynamic_threshold/1e4:.0f}万元")

    end_date = context.previous_date
    TRADE_DAYS_COUNT = 3

    try:
        price_data = get_price(g.fixed_etf_pool, end_date=end_date, count=TRADE_DAYS_COUNT,
                               frequency='daily', fields=['money'], panel=False)

        if price_data is None or price_data.empty:
            log.warning("【固定池过滤】无法获取成交额数据，跳过过滤")
            g.filtered_fixed_pool = g.fixed_etf_pool[:]
            return

        total_money = price_data.groupby('code')['money'].sum()
        avg_daily_money = total_money / TRADE_DAYS_COUNT
        qualified = avg_daily_money[avg_daily_money > dynamic_threshold]
        new_fixed_pool = qualified.index.tolist()

        # 记录被剔除的ETF
        removed = set(g.fixed_etf_pool) - set(new_fixed_pool)
        if removed:
            removed_info = []
            for code in removed:
                try:
                    name = get_security_info(code).display_name
                    money = avg_daily_money.get(code, 0)
                    removed_info.append(f"{name}({code}) {money/1e8:.2f}亿")
                except:
                    removed_info.append(code)
            log.info(f"【固定池过滤】剔除低流动性ETF ({len(removed)}只): {removed_info}")

        g.filtered_fixed_pool = new_fixed_pool

        # 显示保留的ETF
        sorted_qualified = qualified.sort_values(ascending=False)
        kept_info = []
        for code, money in sorted_qualified.items():
            try:
                name = get_security_info(code).display_name
                kept_info.append(f"{name}({code})日均{money/1e8:.2f}亿")
            except:
                kept_info.append(f"{code}日均{money/1e8:.2f}亿")
        log.info(f"【固定池过滤】保留高流动性ETF ({len(new_fixed_pool)}只): {kept_info}")

    except Exception as e:
        log.warning(f"【固定池过滤】异常: {e}")
        g.filtered_fixed_pool = g.fixed_etf_pool[:]


# ==================== 动态池更新 ====================
def update_sector_pool(context):
    """更新行业ETF动态池（每天运行）"""
    log.info("=" * 70)
    log.info("【动态池更新】开始执行")

    # 确保阈值已初始化
    if g.avg_etf_money_threshold is None:
        log.info("【动态池更新】阈值未初始化，立即计算")
        calculate_global_etf_threshold(context)

    # ========== 名称清理常量定义 ==========
    # 基金公司名称列表（按长度降序排序，确保长的先匹配）
    FUND_COMPANIES = sorted(list(set([
        '易方达', '广发', '华夏', '华安', '嘉实', '富国', '招商', '鹏华', '南方', '汇添富', '国泰', '平安',
        '银华', '天弘', '建信', '工银', '华泰柏瑞', '博时', '景顺长城', '景顺', '华宝', '申万菱信', '万家', '中欧',
        '兴证全球', '浙商', '诺安', '前海开源', '泰康', '泰达宏利', '农银汇理', '交银', '东方红', '财通', '华商',
        '国联', '永赢', '金鹰', '德邦', '创金合信', '西部利得', '圆信永丰', '泓德', '汇安', '诺德', '恒生前海',
        '华润元大', '大成', '海富通', '摩根', '华泰', '中信', '中银', '兴全', '国信', '长城', '中金', '浙商证券',
        '东海', '东吴', '浦银安盛', '信达澳亚', '中加', '中航', '中融', '中邮', '中庚', '中信保诚', '中信建投',
        '中银国际', '中银证券', '九泰', '交银施罗德', '光大保德信', '兴银', '农银', '国投瑞银', '国海富兰克林',
        '国联安', '国金', '太平', '方正富邦', '民生加银', '汇丰晋信', '银河', '长信', '长安', '长盛', '长江证券', '鹏扬'
    ])), key=len, reverse=True)

    # 通用噪音词列表（按长度降序排序）
    NOISE_WORDS = sorted(list(set([
        '6666', '8888', '9999', 'A类', 'AH', 'B', 'BS', 'C', 'C类', 'CS', 'DB', 'E', 'E类',
        'ETF', 'ETF基金', 'ETF联接', 'FG', 'G60', 'GF', 'GT', 'HGS', 'LOF', 'LOF基金', 'LOF联接',
        'SG', 'SZ', 'TF', 'TK', 'WJ', 'YH', 'ZS', 'ZZ', '板块', '策略', '产业', '场内', '场外', '低波',
        '基本面', '基金', '精选', '联接', '联接基金', '量化', '龙头', '民企', '民营', '国企', '央企', '智能',
        '全指', '上市开放式', '指基', '指增', '指数', '指数A', '指数C', '指数ETF', '指数基金', '主题', '增强',
        '上海', '黄', '30', '50', '100', '300', '500', '1000', '2000', '大', '新', '四川', '浙江', '湖北',
    ])), key=len, reverse=True)

    # 特别组分类定义（按关键词长度降序排序，确保长词优先匹配）
    SPECIAL_GROUPS = sorted([
        {'name': '创业组', 'keywords': sorted(['创业板', '创业', '创板', '创', '创成长'], key=len, reverse=True),
         'remove_words': sorted(['创业板', '创业', '创板', '创', '创成长'], key=len, reverse=True)},
        {'name': '科创组', 'keywords': sorted(['科创', '科创板', '科综', 'KC', 'K C', '双创', '科创创业', '创创'], key=len, reverse=True),
         'remove_words': sorted(['科创', '科创板', '科综', 'KC', 'K C', '双创', '科创创业', '创创'], key=len, reverse=True)},
        {'name': '香港组', 'keywords': sorted(['恒生', '恒指', '港股', '港股通', 'H股', '香港', '港', 'HKC', 'HK', 'HS', 'H', '中概'], key=len, reverse=True),
         'remove_words': sorted(['恒生', '恒指', '港股', '港股通', 'H股', '香港', '港', 'HKC', 'HK', 'HS', 'H', '中概'], key=len, reverse=True)},
        {'name': '美指组', 'keywords': sorted(['标普', '纳指', '纳斯达克'], key=len, reverse=True),
         'remove_words': sorted(['标普', '纳指', '纳斯达克'], key=len, reverse=True)}
    ], key=lambda x: max(len(kw) for kw in x['keywords']), reverse=True)

    # 排除关键词（宽基指数、债券货币等）
    exclude_keywords = sorted(list(set([
        '300', '500', '1000', '2000', '800', '30', '50', '100', '180', '200',
        '沪深', '中证', '上证', '深证', '深成', 'A50', 'A100', 'A500', '深100',
        '短融', '可转债', '转债', '双债', '利率债', '国债', '地债', '政金债', '国开债', '基准国债', '新综债',
        '信用债', '企业债', '公司债', '城投债', '城投', '美元债', '沪公司债', '科创债', '科债', '科创AAA',
        '自由现金流', '现金流', '现金流E', '现金流基', '现金流TF', '现金流全', '300现金流', '800现金流',
        '货币', '现金', '快线', '快钱', '中银现金', '500现金', '800现金', '现金800', '现金自由', '现金指数',
        '全指现金', '现金全指', 'ESG', 'MSCI', 'MS',
    ])), key=len, reverse=True)

    # ========== 第一步：获取所有ETF ==========
    try:
        all_funds = get_all_securities(['fund']).index.tolist()
    except Exception as e:
        log.warning(f"获取全市场基金列表失败: {e}")
        return

    etf_list = []
    for code in all_funds:
        try:
            info = get_security_info(code)
            if info and info.subtype == 'etf':
                etf_list.append(code)
        except Exception:
            continue
    log.info(f"【动态池更新】全市场ETF总数: {len(etf_list)} 只")

    # ========== 第二步：分类筛选 ==========
    normal_etfs = []           # 普通组ETF
    special_etfs = []          # 特别组ETF
    special_group_map = {}     # 特别组映射
    excluded_count = 0
    special_excluded_count = 0
    special_excluded_list = []
    normal_excluded_list = []

    for code in etf_list:
        try:
            name = get_security_info(code).display_name

            # 检查是否属于特别组
            is_special = False
            matched_group = None
            for group in SPECIAL_GROUPS:
                for kw in group['keywords']:
                    if kw in name:
                        is_special = True
                        matched_group = group['name']
                        break
                if is_special:
                    break

            # 检查是否包含排除关键词
            is_excluded = False
            excluded_match = None
            for k in exclude_keywords:
                if k in name:
                    is_excluded = True
                    excluded_match = k
                    break

            if is_excluded:
                excluded_count += 1
                if is_special:
                    special_excluded_count += 1
                    special_excluded_list.append(f"{name}({code}) - 匹配: {excluded_match}")
                else:
                    normal_excluded_list.append(f"{name}({code}) - 匹配: {excluded_match}")
            else:
                if is_special:
                    special_etfs.append(code)
                    special_group_map[code] = matched_group
                else:
                    normal_etfs.append(code)
        except Exception:
            continue

    # 统计特别组分布
    group_counts = {}
    for code in special_etfs:
        group_name = special_group_map.get(code, '未知')
        group_counts[group_name] = group_counts.get(group_name, 0) + 1

    log.info(f"【动态池更新】特别组分布: {group_counts}")
    log.info(f"【动态池更新】进入特别组: {len(special_etfs)} 只")
    if special_excluded_list:
        log.info(f"【动态池更新】特别组中被排除（宽基/债券等）: {special_excluded_count} 只")
    else:
        log.info(f"【动态池更新】特别组中被排除（宽基/债券等）: 0 只")

    log.info(f"【动态池更新】进入普通组: {len(normal_etfs)} 只")
    if normal_excluded_list:
        log.info(f"【动态池更新】普通组中被排除（宽基/债券等）: {len(normal_excluded_list)} 只")
    else:
        log.info(f"【动态池更新】普通组中被排除（宽基/债券等）: 0 只")
    log.info(f"【动态池更新】共计被排除ETF: {excluded_count} 只")

    # ========== 第三步：流动性筛选 ==========
    end_date = context.previous_date
    TRADE_DAYS_COUNT = 3
    dynamic_threshold = g.avg_etf_money_threshold

    def filter_by_liquidity(etf_codes, group_name):
        """流动性筛选函数"""
        if not etf_codes:
            return [], 0
        try:
            price_data = get_price(etf_codes, end_date=end_date, count=TRADE_DAYS_COUNT,
                                   frequency='daily', fields=['money'], panel=False)
            if price_data is None or price_data.empty:
                log.warning(f"【{group_name}】无法获取成交额数据")
                return [], len(etf_codes)

            total_money = price_data.groupby('code')['money'].sum()
            avg_daily_money = total_money / TRADE_DAYS_COUNT
            qualified = avg_daily_money[avg_daily_money > dynamic_threshold]
            return qualified.sort_values(ascending=False).index.tolist(), len(etf_codes) - len(qualified)
        except Exception as e:
            log.warning(f"【{group_name}】计算成交额异常: {e}")
            return [], len(etf_codes)

    normal_sorted, normal_filtered_out = filter_by_liquidity(normal_etfs, "普通组")
    special_sorted, special_filtered_out = filter_by_liquidity(special_etfs, "特别组")

    log.info(f"【普通组】通过流动性筛选: {len(normal_sorted)} 只，因流动性不足被过滤: {normal_filtered_out} 只")
    log.info(f"【特别组】通过流动性筛选: {len(special_sorted)} 只，因流动性不足被过滤: {special_filtered_out} 只")

    if not normal_sorted and not special_sorted:
        log.info("【动态池更新】没有ETF满足流动性条件")
        g.dynamic_etf_pool = []
        return

    # ========== 第四步：名称清理和行业分组 ==========
    def get_remove_words_for_etf(_, is_special, matched_group_name):
        """获取需要删除的分组词汇"""
        if not is_special:
            return []
        for group in SPECIAL_GROUPS:
            if group['name'] == matched_group_name:
                return group['remove_words']
        return []

    def clean_name(original_name, is_special=False, matched_group_name=None):
        """清理ETF名称"""
        cleaned = original_name
        # 删除基金公司名称
        for company in FUND_COMPANIES:
            cleaned = cleaned.replace(company, '')
        # 删除特别组词汇
        if is_special and matched_group_name:
            for word in get_remove_words_for_etf(original_name, is_special, matched_group_name):
                cleaned = cleaned.replace(word, '')
        # 删除通用噪音词
        for noise in NOISE_WORDS:
            cleaned = cleaned.replace(noise, '')
        return cleaned.strip()

    # 处理普通组
    normal_industry_groups = {}
    normal_cleaned_empty_count = 0

    for code in normal_sorted:
        try:
            original_name = get_security_info(code).display_name
            price_data = get_price([code], end_date=end_date, count=TRADE_DAYS_COUNT,
                                   frequency='daily', fields=['money'], panel=False)
            if price_data is None or price_data.empty:
                continue
            money = price_data['money'].sum() / TRADE_DAYS_COUNT

            cleaned = clean_name(original_name, is_special=False)
            if cleaned == '':
                normal_cleaned_empty_count += 1
                continue

            industry_key = cleaned[:2] if len(cleaned) >= 2 else cleaned
            if industry_key not in normal_industry_groups:
                normal_industry_groups[industry_key] = []
            normal_industry_groups[industry_key].append({
                'code': code, 'original_name': original_name, 'cleaned_name': cleaned,
                'money': money, 'group_type': '普通'
            })
        except Exception:
            continue

    # 处理特别组
    special_industry_groups = {}
    special_cleaned_empty_count = 0

    for code in special_sorted:
        try:
            original_name = get_security_info(code).display_name
            matched_group = special_group_map.get(code, '未知')

            price_data = get_price([code], end_date=end_date, count=TRADE_DAYS_COUNT,
                                   frequency='daily', fields=['money'], panel=False)
            if price_data is None or price_data.empty:
                continue
            money = price_data['money'].sum() / TRADE_DAYS_COUNT

            cleaned = clean_name(original_name, is_special=True, matched_group_name=matched_group)
            if cleaned == '':
                special_cleaned_empty_count += 1
                continue

            industry_key = cleaned[:2] if len(cleaned) >= 2 else cleaned
            group_key = f"{matched_group}_{industry_key}"

            if group_key not in special_industry_groups:
                special_industry_groups[group_key] = []
            special_industry_groups[group_key].append({
                'code': code, 'original_name': original_name, 'cleaned_name': cleaned,
                'money': money, 'group_type': matched_group, 'display_group': matched_group
            })
        except Exception:
            continue

    if normal_cleaned_empty_count > 0 or special_cleaned_empty_count > 0:
        log.info(f"【动态池更新】清理后为空被剔除: 普通组 {normal_cleaned_empty_count} 只，特别组 {special_cleaned_empty_count} 只")

    log.info(f"【动态池更新】普通组行业分类完成: {len(normal_industry_groups)} 个类别")
    log.info(f"【动态池更新】特别组行业分类完成: {len(special_industry_groups)} 个类别")

    # ========== 第五步：每个行业选冠军 ==========
    final_pool_info = []

    # 处理普通组
    for industry_key, items in normal_industry_groups.items():
        sorted_items = sorted(items, key=lambda x: x['money'], reverse=True)
        final_pool_info.append(sorted_items[0])

    # 处理特别组
    for group_key, items in special_industry_groups.items():
        sorted_items = sorted(items, key=lambda x: x['money'], reverse=True)
        final_pool_info.append(sorted_items[0])

    # ========== 第六步：取前100只 ==========
    final_pool_info_sorted = sorted(final_pool_info, key=lambda x: x['money'], reverse=True)
    top_100 = final_pool_info_sorted[:100]
    g.dynamic_etf_pool = [item['code'] for item in top_100]

    log.info(f"【动态池更新】行业去重完成，普通组 {len(normal_industry_groups)} 个行业，特别组 {len(special_industry_groups)} 个行业，合计 {len(normal_industry_groups) + len(special_industry_groups)} 个行业类别，取前100只")

    # 打印最终结果
    etf_display_list = []
    for item in top_100:
        group_flag = f"【{item.get('display_group', '普通')}】" if item.get('group_type') != '普通' else ""
        etf_display_list.append(f"{group_flag}{item['original_name']}({item['code']})日均{item['money']/1e8:.2f}亿 -> [{item['cleaned_name']}]")
    log.info(f"【动态池更新完成】热点资金涌入行业池(前{len(g.dynamic_etf_pool)}只): {etf_display_list}")


# ==================== 合并ETF池 ====================
def daily_merge_etf_pools(context):
    """每日合并固定池和动态池"""
    if not hasattr(g, 'filtered_fixed_pool'):
        g.filtered_fixed_pool = g.fixed_etf_pool[:]

    merged = list(set(g.filtered_fixed_pool + g.dynamic_etf_pool))
    merged.sort()

    log.info("=" * 70)
    log.info("【合并ETF池】开始执行")
    log.info(f"【合并池统计】")
    log.info(f"  - 固定池（过滤后）: {len(g.filtered_fixed_pool)} 只")
    log.info(f"  - 动态池: {len(g.dynamic_etf_pool)} 只")
    log.info(f"  - 合并后去重: {len(merged)} 只")

    g.merged_etf_pool = merged


# ==================== 动量得分计算 ====================
def calculate_and_log_ranked_etfs(context):
    """计算合并池中的标的动量得分"""
    if not hasattr(g, 'merged_etf_pool') or not g.merged_etf_pool:
        log.warning("【动量计算】合并池为空，无法计算")
        g.ranked_etfs_result = []
        return
    final_list = get_final_ranked_etfs(context)
    g.ranked_etfs_result = final_list


def calculate_all_metrics_for_etf(context, etf):
    """计算单个ETF的所有动量指标（包含增强RSI和溢价率）"""
    try:
        etf_name = get_security_name(etf)

        lookback = max(
            g.lookback_days,
            g.short_lookback_days,
            g.rsi_period + g.rsi_lookback_days,
            g.ma_filter_days,
            g.volume_lookback,
            g.rsi_ma_period
        ) + 20

        prices = attribute_history(etf, lookback, '1d', ['close', 'high', 'low'])
        current_data = get_current_data()

        if len(prices) < max(g.lookback_days, g.ma_filter_days):
            return None

        current_price = current_data[etf].last_price
        price_series = np.append(prices["close"].values, current_price)

        # 计算动量得分（加权线性回归）
        recent_price_series = price_series[-(g.lookback_days + 1):]
        y = np.log(recent_price_series)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0
        momentum_score = annualized_returns * r_squared

        # 短期动量
        if len(price_series) >= g.short_lookback_days + 1:
            short_return = price_series[-1] / price_series[-(g.short_lookback_days + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / g.short_lookback_days) - 1
        else:
            short_annualized = -np.inf

        # 均线
        ma_price = np.mean(price_series[-g.ma_filter_days:])
        current_above_ma = current_price >= ma_price

        # 成交量比
        volume_ratio = get_volume_ratio(context, etf, show_detail_log=False)

        # 短期风控（近3日单日跌幅）
        day_ratios = []
        passed_loss_filter = True
        if len(price_series) >= 4:
            day1 = price_series[-1] / price_series[-2]
            day2 = price_series[-2] / price_series[-3]
            day3 = price_series[-3] / price_series[-4]
            day_ratios = [day1, day2, day3]
            if min(day_ratios) < g.loss:
                passed_loss_filter = False

        # ========== 增强的RSI指标 ==========
        passed_rsi_filter = True
        max_recent_rsi = 0
        current_rsi = 0
        rsi_detail = ""
        if g.use_rsi_filter:
            passed_rsi_filter, max_recent_rsi, rsi_detail = check_rsi_filter_enhanced(price_series, current_price, context)
            # 同时记录当前RSI用于日志
            if len(price_series) >= g.rsi_period + 1:
                rsi_vals = calculate_rsi(price_series, g.rsi_period)
                if len(rsi_vals) > 0:
                    current_rsi = rsi_vals[-1]

        # ========== 溢价率计算 ==========
        passed_premium_filter = True
        premium_rate = 0.0
        if g.enable_premium_filter:
            premium_rate, nav, success = get_premium_rate(context, etf)
            if success:
                # 判断是否通过溢价率过滤（溢价率绝对值超过阈值则过滤）
                # 注意：对于LOF，高溢价通常是风险信号；也可根据策略调整方向
                if abs(premium_rate) > g.premium_threshold:
                    passed_premium_filter = False
                else:
                    passed_premium_filter = True
            else:
                # 计算失败时，默认通过过滤（避免因数据问题导致无法选股）
                passed_premium_filter = True

        return {
            'etf': etf,
            'etf_name': etf_name,
            'momentum_score': momentum_score,
            'annualized_returns': annualized_returns,
            'r_squared': r_squared,
            'short_annualized': short_annualized,
            'current_price': current_price,
            'ma_price': ma_price,
            'volume_ratio': volume_ratio,
            'day_ratios': day_ratios,
            'current_rsi': current_rsi,
            'max_recent_rsi': max_recent_rsi,
            'passed_momentum': g.min_score_threshold <= momentum_score <= g.max_score_threshold,
            'passed_short_mom': short_annualized >= g.short_momentum_threshold,
            'passed_r2': r_squared > g.r2_threshold,
            'passed_annual_ret': annualized_returns >= g.min_annualized_return,
            'passed_ma': current_above_ma,
            'passed_volume': volume_ratio is not None and volume_ratio < g.volume_threshold,
            'passed_loss': passed_loss_filter,
            'passed_rsi': passed_rsi_filter,
            'passed_premium': passed_premium_filter,
            'premium_rate': premium_rate,
            'rsi_detail': rsi_detail,
        }
    except Exception as e:
        log.warning(f"计算 {etf} 指标出错: {e}")
        return None


def get_volume_ratio(context, security, lookback_days=None, threshold=None, show_detail_log=True):
    """计算成交量比（当前量/过去N日均量）"""
    if lookback_days is None:
        lookback_days = g.volume_lookback
    try:
        hist_data = attribute_history(security, lookback_days, '1d', ['volume'])
        if hist_data.empty or len(hist_data) < lookback_days:
            return None
        past_n_days_vol = hist_data['volume']
        if past_n_days_vol.isnull().any() or past_n_days_vol.eq(0).any():
            return None
        avg_volume = past_n_days_vol.mean()
        if avg_volume == 0:
            return None
        today = context.current_dt.date()
        df_vol = get_price(security, start_date=today, end_date=context.current_dt, frequency='1m',
                           fields=['volume'], skip_paused=False, fq='pre', panel=False, fill_paused=False)
        if df_vol is None or df_vol.empty:
            return None
        current_volume = df_vol['volume'].sum()
        return current_volume / avg_volume if avg_volume > 0 else 0
    except Exception:
        return None


def calculate_rsi(prices, period=6):
    """计算RSI指标"""
    if len(prices) < period + 1:
        return np.array([])
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    alpha = 2.0 / (period + 1)
    avg_gains = np.zeros(len(deltas))
    avg_losses = np.zeros(len(deltas))
    avg_gains[period - 1] = np.mean(gains[:period])
    avg_losses[period - 1] = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gains[i] = (gains[i] * alpha) + (avg_gains[i - 1] * (1 - alpha))
        avg_losses[i] = (losses[i] * alpha) + (avg_losses[i - 1] * (1 - alpha))
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    full_rsi = np.full(len(prices), np.nan)
    full_rsi[1:] = rsi
    return full_rsi[period:]


def get_final_ranked_etfs(context):
    """主筛选函数，从合并池中选出最终排名ETF（含详细日志，整合增强RSI和溢价率）"""
    all_metrics = []
    etf_set = list(g.merged_etf_pool)

    end_date = context.previous_date

    log.info(f"【动量得分计算】使用合并池，合计{len(etf_set)}只ETF")

    for etf in etf_set:
        try:
            info = get_security_info(etf)
            start_date_raw = info.start_date if info else None
        except Exception:
            start_date_raw = None

        if start_date_raw is None:
            start_date = None
        elif isinstance(start_date_raw, datetime):
            start_date = start_date_raw.date()
        elif isinstance(start_date_raw, date):
            start_date = start_date_raw
        else:
            start_date = None

        if start_date is None or end_date < start_date:
            continue

        current_data = get_current_data()
        if current_data[etf].paused:
            continue

        metrics = calculate_all_metrics_for_etf(context, etf)
        if metrics:
            if metrics['etf'] in {m['etf'] for m in all_metrics}:
                log.warning(f"发现重复ETF数据: {metrics['etf']}，跳过。")
                continue
            all_metrics.append(metrics)

    for item in all_metrics:
        score = item.get('momentum_score')
        if pd.isna(score) or (isinstance(score, float) and np.isnan(score)):
            item['momentum_score'] = float('-inf')

    all_metrics.sort(key=lambda x: x.get('momentum_score', float('-inf')), reverse=True)

    log_lines_step1 = ["", ">>> 第一步：所有ETF按动量得分从大到小排序 <<<"]
    for m in all_metrics:
        def fmt_status(value_str, passed):
            return f"{value_str} {'✅' if passed else '❌'}"

        original_score = m.get('momentum_score')
        if original_score == float('-inf'):
            mom_score_str = "nan"
            mom_passed = False
        else:
            mom_score_str = f"{original_score:.4f}" if not pd.isna(original_score) else "nan"
            mom_passed = m['passed_momentum']

        short_str = f"{m['short_annualized']:.4f}" if not pd.isna(m['short_annualized']) else "nan"
        short = fmt_status(f"短期动量: {short_str}", m['passed_short_mom'])
        r2_str = f"{m['r_squared']:.3f}" if not pd.isna(m['r_squared']) else "nan"
        r2 = fmt_status(f"R²: {r2_str}", m['passed_r2'])
        ann_str = f"{m['annualized_returns']:.2%}" if not pd.isna(m['annualized_returns']) else "nan%"
        ann = fmt_status(f"年化收益率: {ann_str}", m['passed_annual_ret'])
        ma_price_str = f"{m['ma_price']:.2f}" if not pd.isna(m['ma_price']) else "nan"
        ma = fmt_status(f"均线: 当前价{m['current_price']:.2f} vs 均线{ma_price_str}", m['passed_ma'])
        vol_val = f"{m['volume_ratio']:.2f}" if m['volume_ratio'] is not None else "N/A"
        vol = fmt_status(f"成交量比值: {vol_val}", m['passed_volume'])
        min_ratio = min(m['day_ratios']) if m['day_ratios'] else 'N/A'
        loss_val = f"{min_ratio:.4f}" if isinstance(min_ratio, float) and not pd.isna(min_ratio) else str(min_ratio)
        loss = fmt_status(f"短期风控（近3日最低比值）: {loss_val}", m['passed_loss'])
        rsi_status = "✅" if m['passed_rsi'] else "❌"
        rsi_detail = m.get('rsi_detail', '')
        premium_str = f"溢价率: {m['premium_rate']:.2f}% {'✅' if m['passed_premium'] else '❌'}" if g.enable_premium_filter else ""

        line = (
            f"{m['etf']} {m['etf_name']}: "
            f"{fmt_status(f'动量得分: {mom_score_str}', mom_passed)} ，"
            f"{short} ，"
            f"{r2}，"
            f"{ann}，"
            f"{ma}，"
            f"{vol}，"
            f"{loss}，"
            f"RSI过滤: {rsi_status} {rsi_detail}，"
            f"{premium_str}"
        )
        log_lines_step1.append(line)

    # 第二步：应用过滤条件
    filtered_list = apply_filters(all_metrics)
    for item in filtered_list:
        score = item.get('momentum_score')
        if pd.isna(score) or (isinstance(score, float) and np.isnan(score)):
            item['momentum_score'] = float('-inf')
    
    filtered_list.sort(key=lambda x: x.get('momentum_score', float('-inf')), reverse=True)
    
    # 取前10名
    top_10 = filtered_list[:10]
    
    log_lines_step2 = ["", ">>> 第二步：符合全部过滤条件的ETF按动量得分从大到小排序 (前10名) <<<"]
    
    if top_10:
        for i, m in enumerate(top_10):
            def fmt_status(value_str, passed):
                return f"{value_str} {'✅' if passed else '❌'}"

            original_score = m.get('momentum_score')
            if original_score == float('-inf'):
                mom_score_str = "nan"
                mom_passed = False
            else:
                mom_score_str = f"{original_score:.4f}" if not pd.isna(original_score) else "nan"
                mom_passed = m['passed_momentum']

            short_str = f"{m['short_annualized']:.4f}" if not pd.isna(m['short_annualized']) else "nan"
            short = fmt_status(f"短期动量: {short_str}", m['passed_short_mom'])
            r2_str = f"{m['r_squared']:.3f}" if not pd.isna(m['r_squared']) else "nan"
            r2 = fmt_status(f"R²: {r2_str}", m['passed_r2'])
            ann_str = f"{m['annualized_returns']:.2%}" if not pd.isna(m['annualized_returns']) else "nan%"
            ann = fmt_status(f"年化收益率: {ann_str}", m['passed_annual_ret'])
            ma_price_str = f"{m['ma_price']:.2f}" if not pd.isna(m['ma_price']) else "nan"
            ma = fmt_status(f"均线: 当前价{m['current_price']:.2f} vs 均线{ma_price_str}", m['passed_ma'])
            vol_val = f"{m['volume_ratio']:.2f}" if m['volume_ratio'] is not None else "N/A"
            vol = fmt_status(f"成交量比值: {vol_val}", m['passed_volume'])
            min_ratio = min(m['day_ratios']) if m['day_ratios'] else 'N/A'
            loss_val = f"{min_ratio:.4f}" if isinstance(min_ratio, float) and not pd.isna(min_ratio) else str(min_ratio)
            loss = fmt_status(f"短期风控（近3日最低比值）: {loss_val}", m['passed_loss'])
            rsi_status = "✅" if m['passed_rsi'] else "❌"
            rsi_detail = m.get('rsi_detail', '')
            premium_str = f"溢价率: {m['premium_rate']:.2f}% {'✅' if m['passed_premium'] else '❌'}" if g.enable_premium_filter else ""

            line = (
                f"{m['etf']} {m['etf_name']}: "
                f"{fmt_status(f'动量得分: {mom_score_str}', mom_passed)} ，"
                f"{short} ，"
                f"{r2}，"
                f"{ann}，"
                f"{ma}，"
                f"{vol}，"
                f"{loss}，"
                f"RSI过滤: {rsi_status} {rsi_detail}，"
                f"{premium_str}"
            )
            log_lines_step2.append(line)
    else:
        log_lines_step2.append("（无符合条件的ETF）")
        full_log = "\n".join(log_lines_step1 + log_lines_step2)
        log.info(full_log)
        return []
    
    # ========== 第三步：获取参考得分阈值，构建候选池（按动量得分排序） ==========
    if len(top_10) >= g.holdings_num:
        reference_score = top_10[g.holdings_num - 1]['momentum_score']
        score_threshold = reference_score * g.score_threshold_ratio
        log_lines_step3 = [f"", f">>> 第三步：选取动量得分 ≥ 第{g.holdings_num}名 ({top_10[g.holdings_num - 1]['etf_name']}) 得分 {reference_score:.4f} × {g.score_threshold_ratio} = {score_threshold:.4f} 的ETF <<<"]
        
        candidate_pool = [item for item in top_10 if item['momentum_score'] >= score_threshold]
    else:
        log_lines_step3 = [f"", f">>> 第三步：前10名不足{g.holdings_num}只，全部作为候选池 <<<"]
        candidate_pool = top_10[:]

    log_lines_step3.append(f"【候选池】共{len(candidate_pool)}只ETF（按动量得分排序）：")
    for i, item in enumerate(candidate_pool):
        log_lines_step3.append(f"  {i+1}. {item['etf_name']}({item['etf']}) 动量得分: {item['momentum_score']:.4f}")

    # ========== 第四步：结合当前持仓进行调整 ==========
    log_lines_step4 = ["", ">>> 第四步：结合当前持仓进行调整 <<<"]

    current_holdings = [sec for sec, pos in context.portfolio.positions.items() if pos.total_amount > 0]
    log_lines_step4.append(f"当前持仓ETF：{current_holdings}")

    candidate_dict = {item['etf']: item for item in candidate_pool}

    retained = [candidate_dict[etf] for etf in current_holdings if etf in candidate_dict]
    log_lines_step4.append(f"其中存在于候选池中的持仓ETF：{[item['etf'] for item in retained]}")

    if len(retained) >= g.holdings_num:
        retained_sorted = sorted(retained, key=lambda x: x.get('momentum_score', float('-inf')), reverse=True)
        final_result = retained_sorted[:g.holdings_num]
        log_lines_step4.append(f"保留的持仓ETF数量({len(retained)})超过目标持仓数({g.holdings_num})，将从保留的ETF中按动量得分取前{g.holdings_num}只作为最终目标。")
    else:
        need = g.holdings_num - len(retained)
        remaining_pool = [item for item in candidate_pool if item['etf'] not in {r['etf'] for r in retained}]
        additional = remaining_pool[:need]
        final_result = retained + additional
        log_lines_step4.append(f"保留持仓ETF {len(retained)}只，还需补充{need}只。")
        if retained:
            log_lines_step4.append("保留的ETF（按原有顺序）：")
            for item in retained:
                log_lines_step4.append(f"  {item['etf_name']}({item['etf']})")
        if additional:
            log_lines_step4.append("补充的ETF（按动量得分排序）：")
            for i, item in enumerate(additional):
                log_lines_step4.append(f"  {i+1}. {item['etf_name']}({item['etf']}) 动量得分: {item['momentum_score']:.4f}")

    log_lines_step4.append(f"【最终目标】共{len(final_result)}只ETF：")
    for i, item in enumerate(final_result):
        log_lines_step4.append(f"  {i+1}. {item['etf_name']}({item['etf']})")
    log_lines_step4.append("==================================================")

    full_log = "\n".join(log_lines_step1 + log_lines_step2 + log_lines_step3 + log_lines_step4)
    log.info(full_log)

    return final_result


# ==================== 交易执行 ====================
def execute_sell_trades(context):
    """卖出交易逻辑"""
    log.info("========== 卖出操作开始 ==========")
    if is_in_cooldown(context):
        log.info("🔒 当前处于冷却期，跳过轮动逻辑中的卖出操作")
        log.info("========== 卖出操作完成 ==========")
        return

    ranked_etfs = getattr(g, 'ranked_etfs_result', [])
    target_etfs = []

    if ranked_etfs:
        for metrics in ranked_etfs[:g.holdings_num]:
            target_etfs.append(metrics['etf'])
            log.info(f"确定最终目标: {metrics['etf']} {metrics['etf_name']}，得分: {metrics['momentum_score']:.4f}")
    else:
        if check_defensive_etf_available(context):
            target_etfs = [g.defensive_etf]
            etf_name = get_security_name(g.defensive_etf)
            log.info(f"🛡️ 确定最终目标(防御模式): {g.defensive_etf} {etf_name}")
        else:
            log.info("💤 无最终目标(空仓模式)")
            target_etfs = []

    g.target_etfs_list = target_etfs
    current_positions = list(context.portfolio.positions.keys())
    target_set = set(target_etfs)

    sell_count = 0
    for security in current_positions:
        position = context.portfolio.positions[security]
        if position.total_amount > 0 and security not in target_set:
            security_name = get_security_name(security)
            success = smart_order_target_value(security, 0, context)
            if success:
                sell_count += 1
                log.info(f"✅ 已成功卖出: {security} {security_name}")

    log.info(f"本次共计划卖出 {sell_count} 只ETF。")
    log.info("========== 卖出操作完成 ==========")


def execute_buy_trades(context):
    """买入交易逻辑"""
    log.info("========== 买入操作开始 ==========")
    exit_safe_haven_if_cooldown_ends(context)
    if is_in_cooldown(context):
        log.info("🔒 当前处于冷却期，跳过正常买入操作")
        log.info("========== 买入操作完成 ==========")
        return

    target_etfs = g.target_etfs_list
    if not target_etfs:
        log.info("根据计算的结果，今日无目标ETF，保持空仓")
        log.info("========== 买入操作完成 ==========")
        return

    current_positions = set(context.portfolio.positions.keys())
    etfs_to_buy = [etf for etf in target_etfs if etf not in current_positions]
    actual_holding_count = len(current_positions)
    max_buy_count = max(0, g.holdings_num - actual_holding_count)
    num_etfs_to_buy = min(len(etfs_to_buy), max_buy_count)

    if num_etfs_to_buy <= 0:
        log.info(f"当前实际持仓数量({actual_holding_count})已达到或超过目标({g.holdings_num})，无需买入")
        log.info("========== 买入操作完成 ==========")
        return

    etfs_to_buy = etfs_to_buy[:num_etfs_to_buy]
    log.info(f"当前实际持仓: {actual_holding_count}只, 目标持仓: {g.holdings_num}只, 本次计划买入: {num_etfs_to_buy}只")

    available_cash = context.portfolio.available_cash
    allocated_value_per_etf = available_cash // num_etfs_to_buy
    log.info(f"账户可用现金: {available_cash:.2f}, 分配给每只ETF的资金: {allocated_value_per_etf:.2f}")

    if allocated_value_per_etf < g.min_money:
        log.info(f"单只ETF分配金额 {allocated_value_per_etf:.2f} 小于最小交易额 {g.min_money:.2f}，无法买入")
        log.info("========== 买入操作完成 ==========")
        return

    for i, etf in enumerate(etfs_to_buy):
        target_value_for_this_etf = allocated_value_per_etf
        if i == len(etfs_to_buy) - 1 and context.portfolio.available_cash >= g.min_money:
            target_value_for_this_etf = context.portfolio.available_cash

        success = smart_order_target_value(etf, target_value_for_this_etf, context)
        if success:
            log.info(f"✅ ETF {etf} 下单成功")
        else:
            log.info(f"❌ ETF {etf} 下单失败")

    log.info("========== 买入操作完成 ==========")


def smart_order_target_value(security, target_value, context):
    """智能下单（考虑停牌、涨跌停、最小交易额等）"""
    current_data = get_current_data()
    security_name = get_security_name(security)

    if current_data[security].paused:
        log.info(f"{security} {security_name}: 今日停牌，跳过交易")
        return False
    if current_data[security].last_price >= current_data[security].high_limit:
        log.info(f"{security} {security_name}: 当前涨停，跳过交易")
        return False
    if current_data[security].last_price <= current_data[security].low_limit:
        log.info(f"{security} {security_name}: 当前跌停，跳过卖出")
        return False

    current_price = current_data[security].last_price
    if current_price == 0:
        log.info(f"{security} {security_name}: 当前价格为0，跳过交易")
        return False

    target_amount = int(target_value / current_price)
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0:
        target_amount = 100

    current_position = context.portfolio.positions.get(security, None)
    current_amount = current_position.total_amount if current_position else 0
    amount_diff = target_amount - current_amount
    trade_value = abs(amount_diff) * current_price

    if 0 < trade_value < g.min_money:
        log.info(f"{security} {security_name}: 交易金额{trade_value:.2f}小于最小交易额{g.min_money}，跳过")
        return False

    if amount_diff < 0:
        closeable_amount = current_position.closeable_amount if current_position else 0
        if closeable_amount == 0:
            log.info(f"{security} {security_name}: 当天买入不可卖出(T+1)")
            return False
        amount_diff = -min(abs(amount_diff), closeable_amount)

    if amount_diff != 0:
        order_result = order(security, amount_diff)
        if order_result:
            g.positions[security] = target_amount
            if amount_diff > 0:
                log.info(f"📦 买入 {security} {security_name}，数量: {amount_diff}，价格: {current_price:.3f}")
            else:
                log.info(f"📤 卖出 {security} {security_name}，数量: {abs(amount_diff)}，价格: {current_price:.3f}")
            return True
        else:
            log.warning(f"下单失败: {security} {security_name}，数量: {amount_diff}")
            return False
    return False


# ==================== 止损机制 ====================
def minute_level_stop_loss(context):
    """分钟级固定比例止损"""
    if not g.use_fixed_stop_loss:
        return
    if is_in_cooldown(context):
        return

    current_data = get_current_data()
    for security in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[security]
        if position.total_amount <= 0:
            continue
        current_price = current_data[security].last_price
        if current_price <= 0:
            continue
        cost_price = position.avg_cost
        if cost_price <= 0:
            continue
        if current_price <= cost_price * g.fixedStopLossThreshold:
            security_name = get_security_name(security)
            loss_percent = (current_price / cost_price - 1) * 100
            log.info(f"🚨 [分钟级] 固定止损卖出: {security} {security_name}，亏损: {loss_percent:.2f}%")
            success = smart_order_target_value(security, 0, context)
            if success:
                log.info(f"✅ [分钟级] 止损成功: {security} {security_name}")
                enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级固定止损")


def minute_level_pct_stop_loss(context):
    """分钟级当日跌幅止损（基于昨日收盘价）"""
    if not g.use_pct_stop_loss:
        return
    if is_in_cooldown(context):
        return

    current_data = get_current_data()
    for security in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[security]
        if position.total_amount <= 0:
            continue
        try:
            close_series = attribute_history(security, 1, '1d', ['close'], skip_paused=False)
            if len(close_series['close']) == 0:
                continue
            yesterday_close = close_series['close'][-1]
            if yesterday_close <= 0:
                continue
        except Exception:
            continue

        current_price = current_data[security].last_price
        if current_price <= 0:
            continue

        stop_price = yesterday_close * g.pct_stop_loss_threshold
        if current_price <= stop_price:
            security_name = get_security_name(security)
            daily_loss = (current_price / yesterday_close - 1) * 100
            log.info(f"🚨 [分钟级] 当日跌幅止损卖出: {security} {security_name}，跌幅: {daily_loss:.2f}%")
            success = smart_order_target_value(security, 0, context)
            if success:
                log.info(f"✅ [分钟级] 止损成功: {security} {security_name}")
                enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级当日跌幅止损")


# ==================== 辅助函数 ====================
def get_security_name(security):
    """安全获取证券名称"""
    try:
        current_data = get_current_data()
        return current_data[security].name
    except Exception:
        return "未知名称"


def check_defensive_etf_available(context):
    """检查防御性ETF是否可交易"""
    current_data = get_current_data()
    defensive_etf = g.defensive_etf
    if current_data[defensive_etf].paused:
        log.info(f"防御性ETF {defensive_etf} 今日停牌")
        return False
    if current_data[defensive_etf].last_price >= current_data[defensive_etf].high_limit:
        log.info(f"防御性ETF {defensive_etf} 当前涨停")
        return False
    if current_data[defensive_etf].last_price <= current_data[defensive_etf].low_limit:
        log.info(f"防御性ETF {defensive_etf} 当前跌停")
        return False
    return True


# ==================== 冷却期机制 ====================
def is_in_cooldown(context):
    """判断是否在冷却期内"""
    if not g.sell_cooldown_enabled or g.cooldown_end_date is None:
        return False
    return context.current_dt.date() <= g.cooldown_end_date


def set_cooldown(context):
    """设置冷却期结束日期"""
    if g.sell_cooldown_enabled:
        g.cooldown_end_date = context.current_dt.date() + pd.Timedelta(days=g.sell_cooldown_days)
        log.info(f"🔒 触发冷却期，结束日期: {g.cooldown_end_date.strftime('%Y-%m-%d')}")


def enter_safe_haven_and_set_cooldown(context, trigger_reason=""):
    """进入冷却期并买入避险ETF"""
    if not g.sell_cooldown_enabled:
        return

    # 卖出所有持仓
    for security in list(context.portfolio.positions.keys()):
        if security in g.filtered_fixed_pool or security == g.defensive_etf:
            position = context.portfolio.positions[security]
            if position.total_amount > 0:
                success = smart_order_target_value(security, 0, context)
                if success:
                    log.info(f"✅ [冷却期] 卖出持仓: {security}")

    # 买入避险ETF
    total_value = context.portfolio.total_value
    if total_value > g.min_money:
        success = smart_order_target_value(g.safe_haven_etf, total_value * 0.99, context)
        if success:
            log.info(f"🛡️ [冷却期] 买入避险ETF: {g.safe_haven_etf}，金额: {total_value * 0.99:.2f}")

    set_cooldown(context)
    log.info(f"🔒 [冷却期] 已进入冷却期，由 [{trigger_reason}] 触发")


def exit_safe_haven_if_cooldown_ends(context):
    """冷却期结束时卖出避险ETF"""
    if not g.sell_cooldown_enabled or g.cooldown_end_date is None:
        return

    current_date = context.current_dt.date()
    if current_date > g.cooldown_end_date:
        log.info(f"🔓 冷却期结束，当前日期: {current_date.strftime('%Y-%m-%d')}")
        if g.safe_haven_etf in context.portfolio.positions:
            position = context.portfolio.positions[g.safe_haven_etf]
            if position.total_amount > 0:
                success = smart_order_target_value(g.safe_haven_etf, 0, context)
                if success:
                    log.info(f"✅ [冷却期结束] 卖出避险ETF: {g.safe_haven_etf}")
        g.cooldown_end_date = None
        log.info(f"🔄 策略恢复正常运行")


def trade(context):
    pass