import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
def get_all_subject_link(url):
    resp = urllib.request.urlopen(url)
    data = resp.read()
    html_str = data.decode('utf8')
    soup = BeautifulSoup(html_str)
    forms = soup.find_all('form')
    form = forms[1]
    table = form.find_all('table')[0]
    all_tbody = table.find_all('tbody')
    idx = 1
    links = []
    while idx < len(all_tbody):
        tbody = all_tbody[idx]
        idx += 1
        link = tbody.th.span.a.attrs['href']
        if link:
            links.append(link)
    return links

def get_subject_content(url):
    content_html = urllib.request.urlopen(url)
    content_html_str = content_html.read()
    content_html_str = content_html_str.decode('utf8')
    content_html_soup = BeautifulSoup(content_html_str)
    all_tables = content_html_soup.find_all('table')
    table = all_tables[3]
    return table.find_all('table')[1].text

file_idx = 0
for i in range(952):
    link = 'https://www.zgny.net/forum.php?mod=forumdisplay&fid=2&filter=&orderby=lastpost&&page=%d' % i
    try:
        links = get_all_subject_link(link)
    except:
        print('link %s get_all_subject_link fail' % link)
        continue
    for link in links:
        try:
            print(link)
            content = get_subject_content(link)
            if content:
                print(file_idx)
                with open('/home/parallels/data/bbs_doc/%u' % file_idx, 'w') as f:
                    f.write(content)
                file_idx += 1
        except:
            print('link %s get_subject_content fail' % link)
            continue