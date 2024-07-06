#!/usr/bin/env python3
import sys
import getpass
import signal
import time
from getpass import getpass
from datetime import datetime
import urllib.request, urllib.parse, urllib.error
import threading
import webbrowser
import ssl

class Proxy:
    proxy_set = {
        'btech': 22, 'dual': 62, 'diit': 21, 'faculty': 82, 'integrated': 21,
        'mtech': 62, 'phd': 61, 'retfaculty': 82, 'staff': 21, 'irdstaff': 21,
        'mba': 21, 'mdes': 21, 'msc': 21, 'msr': 21, 'pgdip': 21
    }
    google = 'http://www.google.com'

    def __init__(self, username, password, proxy_cat):
        self.username = username
        self.password = password
        self.proxy_cat = proxy_cat
        self.auto_proxy = "http://www.cc.iitd.ernet.in/cgi-bin/proxy." + proxy_cat
        
        # Updated urllib2.build_opener to urllib.request.build_opener
        self.urlopener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl._create_unverified_context()),
            urllib.request.ProxyHandler({'auto_proxy': self.auto_proxy})
        )
        self.proxy_page_address = 'https://proxy' + str(Proxy.proxy_set[proxy_cat]) + '.iitd.ernet.in/cgi-bin/proxy.cgi'
        self.new_session_id()
        self.details()

    def is_connected(self):
        proxies = {'http': 'http://proxy' + str(Proxy.proxy_set[self.proxy_cat]) + '.iitd.ernet.in:3128'}
        try:
            # Python 3 requires a specific opener for proxies, unlike Py2's urllib.urlopen(url, proxies=...)
            proxy_handler = urllib.request.ProxyHandler(proxies)
            opener = urllib.request.build_opener(proxy_handler)
            response = opener.open(Proxy.google).read().decode('utf-8')
        except Exception as e:
            return "Not Connected"
        
        if "<title>IIT Delhi Proxy Login</title>" in response:
            return "Login Page"
        elif "<title>Google</title>" in response:
            return "Google"
        else:
            return "Not Connected"

    def get_session_id(self):
        try:
            response = self.open_page(self.proxy_page_address)
        except Exception as e:
            return None
        
        check_token = 'sessionid" type="hidden" value="'
        try:
            token_index = response.index(check_token) + len(check_token)
        except ValueError:
            return None
            
        sessionid = ""
        for i in range(16):
            sessionid += response[token_index+i]
        return sessionid

    def new_session_id(self):
        self.sessionid = self.get_session_id()
        self.loginform = {'sessionid': self.sessionid, 'action': 'Validate', 'userid': self.username, 'pass': self.password}
        self.logout_form = {'sessionid': self.sessionid, 'action': 'logout', 'logout': 'Log out'}
        self.loggedin_form = {'sessionid': self.sessionid, 'action': 'Refresh'}

    def login(self):
        self.new_session_id()
        response = self.submitform(self.loginform)
        if "Either your userid and/or password does'not match." in response:
            return "Incorrect", response
        elif "You are logged in successfully as " + self.username in response:
            def ref():
                if not self.loggedout:
                    res = self.refresh()
                    print("Refresh", datetime.now())
                    if res == 'Session Expired':
                        print("Session Expired Run Script again")
                    else:
                        self.timer = threading.Timer(60.0, ref)
                        self.timer.daemon = True
                        self.timer.start()
            self.timer = threading.Timer(60.0, ref)
            self.timer.daemon = True
            self.timer.start()
            self.loggedout = False
            return "Success", response
        elif "already logged in" in response:
            return "Already", response
        elif "Session Expired" in response:
            return "Expired", response
        else:
            return "Not Connected", response

    def logout(self):
        self.loggedout = True
        response = self.submitform(self.logout_form)
        if "you have logged out from the IIT Delhi Proxy Service" in response:
            return "Success", response
        elif "Session Expired" in response:
            return "Expired", response
        else:
            return "Failed", response

    def refresh(self):
        response = self.submitform(self.loggedin_form)
        if "You are logged in successfully" in response:
            if "You are logged in successfully as " + self.username in response:
                return "Success", response
            else:
                return "Not Logged In"
        elif "Session Expired" in response:
            return "Expired", response
        else:
            return "Not Connected", response

    def details(self):
        for property, value in vars(self).items(): # iteritems() -> items()
            if VERBOSE:
                print(f"{property}: {value}")

    def submitform(self, form):
        # Python 3: Data must be bytes. urlencode returns string, so we encode it.
        data = urllib.parse.urlencode(form).encode('utf-8')
        req = urllib.request.Request(self.proxy_page_address, data)
        # Python 3: read() returns bytes, decode to string for parsing
        return self.urlopener.open(req).read().decode('utf-8', errors='ignore')

    def open_page(self, address):
        return self.urlopener.open(address).read().decode('utf-8', errors='ignore')

STATUS = 0
RESPONSE = 1
VERBOSE = False

def signal_handler(signal, frame):
    # Added error handling if user var is not yet defined
    try:
        print('\nLogout', user.logout()[STATUS])
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print("\n\nUsage: python3 iitdproxy.py file") 
    else:
        # Read the file
        try:
            with open(sys.argv[1], 'r') as f:
                content = f.readlines()[0].strip().split(' ')
                uname = content[0]
                proxycat = content[1]
        except Exception as e:
            print(f"Error reading auth file: {e}")
            sys.exit(1)
            
        passwd = getpass()
        user = Proxy(username=uname, password=passwd, proxy_cat=proxycat)
        login_status = user.login()[STATUS]
        print('\nLogin', login_status)
        if login_status == "Success":
            signal.pause()