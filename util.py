
import time as t
class time:

    def sec_to_hms(second):
        second = int(second)
        h, m, s = (second // 3600), (second//60 - second//3600*60), (second % 60)
        return h, m, s
    
    def str_hms_delta(start_time, hms=False, rjust=False, join=':'):
        time_delta = t.time() - start_time
        h, m, s = time.sec_to_hms(time_delta)
        if not hms:
            return "{1}{0}{2:02d}{0}{3:02d}".format(join, h, m, s)
        elif rjust: 
            m, s = str(m).rjust(2), str(s).rjust(2)
        
        return str(f"{h}h{join}{m}m{join}{s}s")

    def str_hms(second, hms=False, rjust=False, join=':'):
        h, m, s = time.sec_to_hms(second)
        if not hms:
            return "{1}{0}{2:02d}{0}{3:02d}".format(join, h, m, s)
        elif rjust: 
            m, s = str(m).rjust(2), str(s).rjust(2)
        
        return str(f"{h}h{join}{m}m{join}{s}s")

import datetime as dt
from selenium import webdriver
def alerm(webdriver_path=r'C:\Users\danal\Documents\programing\chromedriver.exe', loading_sec=7):
    now = dt.datetime.today()

    if int(now.strftime('%S')) < 60 - loading_sec:
        alarm_time = now + dt.timedelta(minutes=1)
    else:
        alarm_time = now + dt.timedelta(minutes=2)

    alarm_time = alarm_time.strftime('%X')
    driver = webdriver.Chrome(webdriver_path)
    driver.get(f'https://vclock.kr/#time={alarm_time}&title=%EC%95%8C%EB%9E%8C&sound=musicbox&loop=1')
    driver.find_element_by_xpath('//*[@id="pnl-main"]').click()
    input('\033[1mPress Enter\033[0m')

import sys
class bcolors:
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ORANGE = '\u001b[38;5;208m'

    c = [HEADER, OKBLUE, OKCYAN, OKGREEN, WARNING, FAIL, ENDC, BOLD, UNDERLINE, ORANGE]

    def test():
        for color in bcolors.c:
            print(color + "Warning: No active frommets remain. Continue?" + bcolors.ENDC)

    def according_to_score(x):
        if x < 1:
            return bcolors.ENDC
        elif 1 <= x < 20:
            return bcolors.FAIL
        elif 20 <= x < 40:
            return bcolors.ORANGE
        elif 40 <= x < 60:
            return bcolors.WARNING
        elif 60 <= x < 80:
            return bcolors.OKGREEN
        elif 80 <= x < 95:
            return bcolors.OKBLUE
        elif 95 <= x < 99:
            return bcolors.HEADER
        else:
            return bcolors.BOLD
    
    def according_to_chance(x):
        if x < 0.5:
            return bcolors.ENDC
        elif 0.5 <= x < 5:
            return bcolors.FAIL
        elif 5 <= x < 20:
            return bcolors.ORANGE
        elif 20 <= x < 50:
            return bcolors.WARNING
        elif 50 <= x < 80:
            return bcolors.OKGREEN
        elif 80 <= x < 95:
            return bcolors.OKBLUE
        elif 95 <= x < 99:
            return bcolors.HEADER
        else:
            return bcolors.BOLD

    def ANSI_codes():
        for i in range(0, 16):
            for j in range(0, 16):
                code = str(i * 16 + j)
                sys.stdout.write(u"\u001b[38;5;" + code + "m" + code.ljust(4))
            print(u"\u001b[0m")


import logging
def __get_logger():
    """로거 인스턴스 반환
    """

    __logger = logging.getLogger('logger')

    # 로그 포멧 정의
    formatter = logging.Formatter(
        '\n"%(pathname)s", line %(lineno)d, in %(module)s\n%(levelname)-8s: %(message)s')
    # 스트림 핸들러 정의
    stream_handler = logging.StreamHandler()
    # 각 핸들러에 포멧 지정
    stream_handler.setFormatter(formatter)
    # 로거 인스턴스에 핸들러 삽입
    __logger.addHandler(stream_handler)
    # 로그 레벨 정의
    __logger.setLevel(logging.DEBUG)

    return __logger
