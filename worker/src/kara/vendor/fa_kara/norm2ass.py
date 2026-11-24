import re

from utils_basic import parse_time_to_hundredths, int2asstime

newnums = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩',
           '⑪', '⑫', '⑬', '⑭', '⑮', '⑯', '⑰', '⑱', '⑲', '⑳',
           '㉑', '㉒', '㉓', '㉔', '㉕', '㉖', '㉗', '㉘', '㉙', '㉚']

def process_norm2assV1(struc, pretime = 20, posttime = 20):
    '模型输出的实际时值，不再维护'
    result = ''
    for i in range(len(struc)):
        if not result or result[-1]=='\n':
            asstxt = ''
            nowtime = starttime = parse_time_to_hundredths([itemd for itemd in struc[i:] if itemd['type'] != 0][0]['start']) - pretime
        item = struc[i]
        if item['type'] == 0 and item['orig'] == '\n':
            try:
                nowtime = parse_time_to_hundredths(item['start'])
            except:
                pass
            finally:
                endtime = nowtime + posttime
                asstxt = 'Dialogue: 0,'+int2asstime(starttime)+','+int2asstime(endtime)+r',Default,,0,0,0,karaoke,'+asstxt+r'{\k'+str(posttime)+'}'
                result += asstxt+'\n'
        elif 'start' not in item:
            asstxt += item['orig']
        else:
            item_kbefore = parse_time_to_hundredths(item['start']) - nowtime
            if item_kbefore!=0:
                asstxt += r'{\k'+str(item_kbefore)+'}'
            item_kdur = parse_time_to_hundredths(item['end']) - parse_time_to_hundredths(item['start'])
            asstxt += r'{\k'+str(item_kdur)+'}'
            if item['type'] == 2:
                asstxt += ('#|' if item['orig']=='' else item['orig'] + '|<') + item['ruby']
            else:
                asstxt += item['orig']
            nowtime = parse_time_to_hundredths(item['end'])
    return result

def process_norm2assV2(struc, pretime = 20, posttime = 20):
    '仿NicokaraMaker.lrc时值'
    result = ''
    starttime = nowtime = None
    asstxt = ''
    i = 0
    while i < len(struc):
        item = struc[i]
        if not starttime:
            try:
                starttime = parse_time_to_hundredths(item['start']) - pretime
                nowtime = parse_time_to_hundredths(item['start'])
            except:
                asstxt += item['orig']
                i += 1
                continue
        if item['type'] == 0 and item['orig'] == '\n':
            try:
                nowtime = parse_time_to_hundredths(item['start'])
            except:
                pass
            finally:
                endtime = nowtime + posttime
                if asstxt:
                    if asstxt[0] not in ['{'] + newnums: 
                        asstxt = r'{\k0}' + asstxt
                    if asstxt[0] in newnums and asstxt[1] != '{':
                        asstxt = asstxt[0] + r'{\k0}' + asstxt[1:]
                result += 'Dialogue: 0,'+int2asstime(starttime)+','+int2asstime(endtime)+r',Default,,0,0,0,karaoke,' \
                    +r'{\k'+str(pretime)+'}'+asstxt+r'{\k'+str(posttime)+'}'+'\n'
                starttime = nowtime = None
                asstxt = ''
        elif item['type'] == 0 and 'start' not in item:
            if struc[i-1].get('type') in (1,3,4,5) and item.get('orig') not in [' ','　']+newnums:
                asstxt += item['orig']
            # elif struc[i-1].get('type') == 2 and item.get('orig') not in [' ','　']+newnums:
            #     asstxt += r'{\k0}' + item['orig']
            else:
                zero_str = item.get('orig')
                while True:
                    if struc[i+1].get('orig') == '\n':
                        item_kdur = 0
                        break
                    if struc[i+1].get('start'):
                        if nowtime:
                            item_kdur = parse_time_to_hundredths(struc[i+1]['start']) - nowtime
                        else:
                            item_kdur = 0
                        nowtime = parse_time_to_hundredths(struc[i+1]['start'])
                        break
                    zero_str += struc[i+1].get('orig')
                    i += 1
                asstxt += r'{\k'+str(item_kdur)+'}' + zero_str
        elif item['type'] == 2 and 'start' not in item:
            asstxt += item.get('ruby', '')
        else:
            # Missing-timestamp guard: an aligned token can arrive without a
            # timestamp when the aligner drops it; emit plain text instead of
            # crashing the whole document.
            if 'start' not in item:
                asstxt += item.get('orig', '')
                i += 1
                continue
            # Use the aligner's token end. Extending a token to the next token's
            # start makes karaoke freeze on a word during pauses, then race
            # through the remaining words. Untimed spaces below represent gaps.
            item_start = parse_time_to_hundredths(item['start'])
            item_end = parse_time_to_hundredths(item['end'])
            if nowtime is not None and item_start > nowtime:
                asstxt += r'{\k'+str(item_start - nowtime)+'}'
            item_kdur = max(0, item_end - item_start)
            nowtime = item_end
            asstxt += r'{\k'+str(item_kdur)+'}'
            if item['type'] == 2:
                asstxt += ('#|' if item['orig']=='' else item['orig'] + '|<') + item['ruby']
            else:
                asstxt += item['orig']
        i += 1
    return result


if __name__=='__main__':
    input_struc = [{'orig': '歌',
  'type': 2,
  'ruby': 'う',
  'pron': 'u',
  'start': '[00:01:38]',
  'end': '[00:01:40]'},
 {'orig': '',
  'type': 2,
  'ruby': 'た',
  'pron': 'ta',
  'start': '[00:01:54]',
  'end': '[00:01:64]'},
 {'orig': 'え',
  'type': 3,
  'pron': 'e',
  'start': '[00:03:12]',
  'end': '[00:03:14]'},
 {'orig': '\u3000', 'type': 0},
 {'orig': '踊',
  'type': 2,
  'ruby': 'お',
  'pron': 'o',
  'start': '[00:03:35]',
  'end': '[00:03:37]'},
 {'orig': '',
  'type': 2,
  'ruby': 'ど',
  'pron': 'do',
  'start': '[00:04:13]',
  'end': '[00:04:24]'},
 {'orig': 'れ',
  'type': 3,
  'pron': 're',
  'start': '[00:10:24]',
  'end': '[00:10:54]'},
 {'orig': '\n', 'type': 0},
 {'orig': '祈',
  'type': 2,
  'ruby': 'い',
  'pron': 'i',
  'start': '[00:11:17]',
  'end': '[00:11:20]'},
 {'orig': '',
  'type': 2,
  'ruby': 'の',
  'pron': 'no',
  'start': '[00:11:33]',
  'end': '[00:11:40]'},
 {'orig': 'れ',
  'type': 3,
  'pron': 're',
  'start': '[00:11:75]',
  'end': '[00:11:85]'},
 {'orig': '\u3000', 'type': 0},
 {'orig': '届',
  'type': 2,
  'ruby': 'と',
  'pron': 'to',
  'start': '[00:11:96]',
  'end': '[00:12:06]'},
 {'orig': '',
  'type': 2,
  'ruby': 'ど',
  'pron': 'do',
  'start': '[00:12:17]',
  'end': '[00:12:72]'},
 {'orig': 'け',
  'type': 3,
  'pron': 'ke',
  'start': '[00:12:83]',
  'end': '[00:12:93]'},
 {'orig': '\n', 'type': 0}]
    print(process_norm2assV2(input_struc))