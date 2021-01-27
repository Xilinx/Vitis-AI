
// Copyright 2012 Google Inc. All rights reserved.
(function(w,g){w[g]=w[g]||{};w[g].e=function(s){return eval(s);};})(window,'google_tag_manager');(function(){

var data = {
"resource": {
  "version":"46",
  
  "macros":[{
      "function":"__aev",
      "vtp_setDefaultValue":false,
      "vtp_varType":"ELEMENT"
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){return ",["escape",["macro",0],8,16],".parentElement.getAttribute(\"href\")})();"]
    },{
      "function":"__u",
      "vtp_component":"URL",
      "vtp_enableMultiQueryKeys":false,
      "vtp_enableIgnoreEmptyQueryParam":false
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=",["escape",["macro",2],8,16],".split(\"\/\");a=a.pop();return a=a.substr(0,a.lastIndexOf(\".\"))||a})();"]
    },{
      "function":"__aev",
      "vtp_setDefaultValue":false,
      "vtp_varType":"URL",
      "vtp_component":"PATH",
      "vtp_defaultPages":["list"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=",["escape",["macro",4],8,16],".split(\"\/\");a=a.pop();return-1\u003Ca.indexOf(\".\")?a:\"n\/a\"})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){return ",["escape",["macro",0],8,16],".parentElement.getAttribute(\"data-anchor\")})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){return ",["escape",["macro",0],8,16],".getAttribute(\"data-anchor\")})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=document.getElementsByTagName(\"a\");for(i=0;i\u003Ca.length;i++)if(a[i].getAttribute(\"data-anchor\")\u0026\u0026\"active\"==a[i].parentElement.getAttribute(\"class\"))return a[i].getAttribute(\"data-anchor\")})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=",["escape",["macro",0],8,16],".closest(\".CoveoSearchInterface\");return(a=a.querySelector('.magic-box-input .magic-box-underlay span[data-id\\x3d\"Any\"]'))?a.getAttribute(\"data-value\"):\"\"})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=",["escape",["macro",0],8,16],".form;return a.find('input[type\\x3d\"radio\"][name\\x3d\"feedbackResults\"]:checked').val()})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){return ",["escape",["macro",0],8,16],".getAttribute(\"data-analytics\")})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){return ",["escape",["macro",0],8,16],".getAttribute(\"data-function\")})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){return\"headerSearchBox\"==",["escape",["macro",0],8,16],".closest(\".CoveoSearchInterface\").getAttribute(\"id\")?\"Header\":\"Non-header\"})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=$(\"button.btn.selected\").text().trim();return a})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=",["escape",["macro",0],8,16],".closest(\".at-share-btn\").getAttribute(\"class\");return a.endsWith(\"at-svc-linkedin\")?\"LinkedIn\":a.endsWith(\"at-svc-twitter\")?\"Twitter\":a.endsWith(\"at-svc-facebook\")?\"Facebook\":a.endsWith(\"at-svc-google_plusone_share\")?\"Google Plus\":a.endsWith(\"at-svc-email\")?\"Email\":a.endsWith(\"at-svc-wechat\")?\"We Chat\":a.endsWith(\"at-svc-tencentqq\")?\"Tencent QQ\":a.endsWith(\"at-svc-sinaweibo\")?\"Sina Weibo\":\"None\"})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=",["escape",["macro",0],8,16],".closest(\".CoveoSearchInterface\"),b=a.getAttribute(\"id\");return b\u0026\u0026(a.querySelector(\".supportSearch\")||a.querySelector(\".partnerDropdown\"))?\"Xilinx-Support-Partner\":\"Not-Support-Partner\"})();"]
    },{
      "function":"__v",
      "vtp_name":"gtm.element",
      "vtp_dataLayerVersion":1
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var b=",["escape",["macro",17],8,16],".closest(\"a\");if(b){var a=b.getAttribute(\"id\");var c=b.getAttribute(\"class\")}return\"video-info\"!==a\u0026\u0026\"video-links\"!==a\u0026\u0026\"video-documents\"!==a||\"disableHover\"!==c?null:a})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a,b=",["escape",["macro",0],8,16],".closest(\".CoveoSearchInterface\"),c=b.querySelector(\".btn.dropdown-toggle\");c\u0026\u0026(a=b.querySelector(\"li[data-label\\x3d'\"+c.innerText+\"']\").getAttribute(\"data-action-link\"));a||(a=b.querySelector(\".CoveoSearchbox\").getAttribute(\"data-action-link\"));a||(a=\"searchComponent\");return a})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var B=\"UA-11440126-15\",x=\"auto\",L=null;!function(){function h(a){Ua.set(a)}function q(a){}function x(){}function B(){}function F(a){}function L(a){}function Va(a){}function r(a,b,c,e){b[a]=function(){try{return e\u0026\u0026h(e),c.apply(this,arguments)}catch(u){throw u;}}}function da(a,b,c){\"none\"==b\u0026\u0026(b=\"\");var e=[],u=G(a);a=\"__utma\"==a?6:2;for(var d=0;d\u003Cu.length;d++){var g=(\"\"+u[d]).split(\".\");g.length\u003E=a\u0026\u0026e.push({hash:g[0],R:u[d],O:g})}if(0!=e.length)return 1==e.length?e[0]:ea(b,e)||ea(c,e)||\nea(null,e)||e[0]}function ea(a,b){var c;null==a?c=a=1:(c=M(a),a=M(0==a.indexOf(\".\")?a.substring(1):\".\"+a));for(var e=0;e\u003Cb.length;e++)if(b[e].hash==c||b[e].hash==a)return b[e]}function Wa(a){var b=a.get(v);if(a.get(z))return a=a.get(H),c=C(a+b,0),\"_ga\\x3d2.\"+N(c+\".\"+a+\"-\"+b);var c=C(b,0);return\"_ga\\x3d1.\"+N(c+\".\"+b)}function C(a,b){var c=new Date,e=p.navigator,d=e.plugins||[];a=[a,e.userAgent,c.getTimezoneOffset(),c.getYear(),c.getDate(),c.getHours(),c.getMinutes()+b];for(b=0;b\u003Cd.length;++b)a.push(d[b].description);\nreturn M(a.join(\".\"))}function ta(a,b){if(b==k.location.hostname)return!1;for(var c=0;c\u003Ca.length;c++)if(a[c]instanceof RegExp){if(a[c].test(b))return!0}else if(0\u003C=b.indexOf(a[c]))return!0;return!1}function M(a){var b,c=1;if(a)for(c=0,b=a.length-1;0\u003C=b;b--){var e=a.charCodeAt(b);c=(c\u003C\u003C6\u0026268435455)+e+(e\u003C\u003C14);e=266338304\u0026c;c=0!=e?c^e\u003E\u003E21:c}return c}var R=function(a){this.w=a||[]};R.prototype.set=function(a){this.w[a]=!0};R.prototype.encode=function(){for(var a=[],b=0;b\u003Cthis.w.length;b++)this.w[b]\u0026\u0026(a[Math.floor(b\/\n6)]^=1\u003C\u003Cb%6);for(b=0;b\u003Ca.length;b++)a[b]=\"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_\".charAt(a[b]||0);return a.join(\"\")+\"~\"};var Ua=new R,Xa=function(a){return a?a.replace(\/^[\\s\\xa0]+|[\\s\\xa0]+$\/g,\"\"):\"\"},ua=function(){for(var a=p.navigator.userAgent+(k.cookie?k.cookie:\"\")+(k.referrer?k.referrer:\"\"),b=a.length,c=p.history.length;0\u003Cc;)a+=c--^b++;return[S()^2147483647\u0026M(a),Math.round((new Date).getTime()\/1E3)].join(\".\")},Ya=function(){},N=function(a){return encodeURIComponent instanceof\nFunction?encodeURIComponent(a):(h(28),a)},T=function(a,b,c,e){try{a.addEventListener?a.addEventListener(b,c,!!e):a.attachEvent\u0026\u0026a.attachEvent(\"on\"+b,c)}catch(u){h(27)}},va=\/^[\\w\\-:\/.?=\u0026%!]+$\/,U=function(){var a=\"\"+k.location.hostname;return 0==a.indexOf(\"www.\")?a.substring(4):a},wa=function(a,b){if(1==b.length\u0026\u0026null!=b[0]\u0026\u0026\"object\"==typeof b[0])return b[0];for(var c={},e=Math.min(a.length+1,b.length),d=0;d\u003Ce;d++){if(\"object\"==typeof b[d]){for(var f in b[d])b[d].hasOwnProperty(f)\u0026\u0026(c[f]=b[d][f]);break}d\u003C\na.length\u0026\u0026(c[a[d]]=b[d])}return c},D=function(){this.keys=[];this.values={};this.m={}};D.prototype.set=function(a,b,c){this.keys.push(a);c?this.m[\":\"+a]=b:this.values[\":\"+a]=b};D.prototype.get=function(a){return this.m.hasOwnProperty(\":\"+a)?this.m[\":\"+a]:this.values[\":\"+a]};D.prototype.map=function(a){for(var b=0;b\u003Cthis.keys.length;b++){var c=this.keys[b],e=this.get(c);e\u0026\u0026a(c,e)}};var p=window,k=document,fa=window,G=function(a){var b=[],c=k.cookie.split(\";\");a=new RegExp(\"^\\\\s*\"+a+\"\\x3d\\\\s*(.*?)\\\\s*$\");\nfor(var e=0;e\u003Cc.length;e++){var d=c[e].match(a);d\u0026\u0026b.push(d[1])}return b},V=function(a,b,c,e,d,f){a:{var u=fa._gaUserPrefs;if(u\u0026\u0026u.ioo\u0026\u0026u.ioo()||d\u0026\u0026!0===fa[\"ga-disable-\"+d])var g=!0;else{try{var l=fa.external;if(l\u0026\u0026l._gaUserPrefs\u0026\u0026\"oo\"==l._gaUserPrefs){g=!0;break a}}catch(Vb){}g=!1}}if(g||Za.test(k.location.hostname)||\"\/\"==c\u0026\u0026$a.test(e))return!1;if(b\u0026\u00261200\u003Cb.length\u0026\u0026(b=b.substring(0,1200),h(24)),c=a+\"\\x3d\"+b+\"; path\\x3d\"+c+\"; \",f\u0026\u0026(c+=\"expires\\x3d\"+(new Date((new Date).getTime()+f)).toGMTString()+\n\"; \"),e\u0026\u0026\"none\"!=e\u0026\u0026(c+=\"domain\\x3d\"+e+\";\"),e=k.cookie,k.cookie=c,!(e=e!=k.cookie))a:{a=G(a);for(e=0;e\u003Ca.length;e++)if(b==a[e]){e=!0;break a}e=!1}return e},ha=function(a){return N(a).replace(\/\\(\/g,\"%28\").replace(\/\\)\/g,\"%29\")},$a=\/^(www\\.)?google(\\.com?)?(\\.[a-z]{2})?$\/,Za=\/(^|\\.)doubleclick\\.net$\/i,xa=function(){this.M=[]};xa.prototype.add=function(a){};var S=function(){return Math.round(2147483647*Math.random())},W=function(){this.data=new D},X=new D,ia=[];W.prototype.get=function(a){var b=ya(a),\nc=this.data.get(a);return b\u0026\u0026void 0==c\u0026\u0026(c=\"function\"==typeof b.defaultValue?b.defaultValue():b.defaultValue),b\u0026\u0026b.Z?b.Z(this,a,c):c};var m=function(a,b){return a=a.get(b),void 0==a?\"\":\"\"+a},ja=function(a,b){return a=a.get(b),void 0==a||\"\"===a?0:1*a};W.prototype.set=function(a,b,c){if(a)if(\"object\"==typeof a)for(var e in a)a.hasOwnProperty(e)\u0026\u0026za(this,e,a[e],c);else za(this,a,b,c)};var za=function(a,b,c,e){if(void 0!=c)switch(b){case A:ab.test(c)}var d=ya(b);d\u0026\u0026d.o?d.o(a,b,c,e):a.data.set(b,c,e)},\nO=function(a,b,c,e,d){this.name=a;this.F=b;this.Z=e;this.o=d;this.defaultValue=c},ya=function(a){var b=X.get(a);if(!b)for(var c=0;c\u003Cia.length;c++){var e=ia[c],d=e[0].exec(a);if(d){b=e[1](d);X.set(b.name,b);break}}return b},bb=function(a){var b;return X.map(function(c,e){e.F==a\u0026\u0026(b=e)}),b\u0026\u0026b.name},d=function(a,b,c,e,d){return a=new O(a,b,c,e,d),X.set(a.name,a),a.name},Y=function(a,b){ia.push([new RegExp(\"^\"+a+\"$\"),b])},l=function(a,b,c){return d(a,b,c,void 0,Aa)},Aa=function(){},P=\"slga\",Z=!1,cb=l(\"apiVersion\",\n\"v\"),db=l(\"clientVersion\",\"_v\");d(\"anonymizeIp\",\"aip\");var eb=d(\"adSenseId\",\"a\"),Ba=d(\"hitType\",\"t\");d(\"hitCallback\");d(\"hitPayload\");d(\"nonInteraction\",\"ni\");d(\"currencyCode\",\"cu\");d(\"dataSource\",\"ds\");d(\"useBeacon\",void 0,!1);d(\"transport\");d(\"sessionControl\",\"sc\",\"\");d(\"sessionGroup\",\"sg\");d(\"queueTime\",\"qt\");d(\"_s\",\"_s\");d(\"screenName\",\"cd\");var fb=(d(\"location\",\"dl\",\"\"),d(\"referrer\",\"dr\"),d(\"page\",\"dp\",\"\"));d(\"hostname\",\"dh\");d(\"language\",\"ul\");d(\"encoding\",\"de\");d(\"title\",\"dt\",function(){return k.title||\nvoid 0});Y(\"contentGroup([0-9]+)\",function(a){return new O(a[0],\"cg\"+a[1])});d(\"screenColors\",\"sd\");d(\"screenResolution\",\"sr\");d(\"viewportSize\",\"vp\");d(\"javaEnabled\",\"je\");d(\"flashVersion\",\"fl\");d(\"campaignId\",\"ci\");d(\"campaignName\",\"cn\");d(\"campaignSource\",\"cs\");d(\"campaignMedium\",\"cm\");d(\"campaignKeyword\",\"ck\");d(\"campaignContent\",\"cc\");var gb=d(\"eventCategory\",\"ec\"),hb=d(\"eventAction\",\"ea\"),ib=d(\"eventLabel\",\"el\"),jb=d(\"eventValue\",\"ev\"),kb=d(\"socialNetwork\",\"sn\"),lb=d(\"socialAction\",\"sa\"),mb=\nd(\"socialTarget\",\"st\"),nb=(d(\"l1\",\"plt\"),d(\"l2\",\"pdt\"),d(\"l3\",\"dns\"),d(\"l4\",\"rrt\"),d(\"l5\",\"srt\"),d(\"l6\",\"tcp\"),d(\"l7\",\"dit\"),d(\"l8\",\"clt\"),d(\"timingCategory\",\"utc\")),ob=d(\"timingVar\",\"utv\"),pb=d(\"timingLabel\",\"utl\"),qb=d(\"timingValue\",\"utt\");d(\"appName\",\"an\");d(\"appVersion\",\"av\",\"\");d(\"appId\",\"aid\",\"\");d(\"appInstallerId\",\"aiid\",\"\");d(\"exDescription\",\"exd\");d(\"exFatal\",\"exf\");var rb=(d(\"expId\",\"xid\"),d(\"expVar\",\"xvar\"),d(\"exp\",\"exp\"),d(\"_utma\",\"_utma\")),sb=d(\"_utmz\",\"_utmz\"),tb=d(\"_utmht\",\"_utmht\");\nd(\"_hc\",void 0,0);d(\"_ti\",void 0,0);d(\"_to\",void 0,20);Y(\"dimension([0-9]+)\",function(a){return new O(a[0],\"cd\"+a[1])});Y(\"metric([0-9]+)\",function(a){return new O(a[0],\"cm\"+a[1])});d(\"linkerParam\",void 0,void 0,Wa,Aa);d(\"usage\",\"_u\");var Ca=d(\"_um\");d(\"forceSSL\",void 0,void 0,function(){return Z},function(a,b,c){h(34);Z=!!c});var ub=d(\"_j1\",\"jid\"),vb=d(\"_j2\",\"gjid\");Y(\"\\\\\\x26(.*)\",function(a){var b=new O(a[0],a[1]),c=bb(a[0].substring(1));return c\u0026\u0026(b.Z=function(a){return a.get(c)},b.o=function(a,\nb,d,g){a.set(c,d,g)},b.F=void 0),b});var wb=l(\"_oot\"),xb=d(\"previewTask\"),yb=d(\"checkProtocolTask\"),zb=d(\"validationTask\"),Ab=d(\"checkStorageTask\"),Bb=d(\"historyImportTask\"),Cb=(d(\"samplerTask\"),d(\"_rlt\"));d(\"buildHitTask\");d(\"sendHitTask\");var Db=(d(\"ceTask\"),d(\"devIdTask\")),Eb=(d(\"timingTask\"),d(\"displayFeaturesTask\")),y=l(\"name\"),v=l(\"clientId\",\"cid\"),Da=l(\"clientIdTime\"),Ea=d(\"userId\",\"uid\"),A=l(\"trackingId\",\"tid\"),Q=l(\"cookieName\",void 0,\"_ga\"),t=l(\"cookieDomain\"),E=l(\"cookiePath\",void 0,\"\/\"),\nka=l(\"cookieExpires\",void 0,63072E3),aa=l(\"legacyCookieDomain\"),la=l(\"legacyHistoryImport\",void 0,!0),I=l(\"storage\",void 0,\"cookie\"),ma=l(\"allowLinker\",void 0,!1),na=l(\"allowAnchor\",void 0,!0),Fa=l(\"sampleRate\",\"sf\",100),oa=l(\"siteSpeedSampleRate\",void 0,1),Ga=l(\"alwaysSendReferrer\",void 0,!1),H=l(\"_gid\",\"_gid\"),z=l(\"_ge\"),pa=l(\"_gcn\"),Fb=d(\"transportUrl\"),Gb=d(\"_r\",\"_r\"),qa=function(a,b,c){this.V=a;this.fa=b;this.$=!1;this.oa=c;this.ea=1},ra=function(a,b,c){if(a.fa\u0026\u0026a.$)return 0;if(a.$=!0,b){if(a.oa\u0026\u0026\nja(b,a.oa))return ja(b,a.oa);if(0==b.get(oa))return 0}return 0==a.V?0:(void 0===c\u0026\u0026(c=void 0),0==c%a.V?Math.floor(c\/a.V)%a.ea+1:0)},J=!1,Ia=function(a){\"cookie\"==m(a,I)\u0026\u0026(Ha(a,v,Q),a.get(z)\u0026\u0026Ha(a,H,pa,864E5))},Ha=function(a,b,c,e){var d=Ja(a,b);if(d){c=m(a,c);b=Ka(m(a,E));var f=sa(m(a,t));e=e||1E3*ja(a,ka);var g=m(a,A);if(\"auto\"!=f)V(c,d,b,f,g,e)\u0026\u0026(J=!0);else{h(32);var w;if(d=[],f=U().split(\".\"),4!=f.length||(w=f[f.length-1],parseInt(w,10)!=w)){for(w=f.length-2;0\u003C=w;w--)d.push(f.slice(w).join(\".\"));\nd.push(\"none\");w=d}else w=[\"none\"];for(var k=0;k\u003Cw.length;k++)if(f=w[k],a.data.set(t,f),d=Ja(a,v),V(c,d,b,f,g,e))return void(J=!0);a.data.set(t,\"auto\")}}else a.get(z)||h(54)},Hb=function(a){if(\"cookie\"==m(a,I)\u0026\u0026!J\u0026\u0026(Ia(a),!J))throw\"abort\";},Ib=function(a){if(a.get(la)){var b=m(a,t),c=m(a,aa)||U(),e=da(\"__utma\",c,b);e\u0026\u0026(h(19),a.set(tb,(new Date).getTime(),!0),a.set(rb,e.R),(b=da(\"__utmz\",c,b))\u0026\u0026e.hash==b.hash\u0026\u0026a.set(sb,b.R))}},Ja=function(a,b){b=ha(m(a,b));var c=sa(m(a,t)).split(\".\").length;return a=\nLa(m(a,E)),1\u003Ca\u0026\u0026(c+=\"-\"+a),b?[\"GA1\",c,b].join(\".\"):\"\"},Na=function(a,b){if(b\u0026\u0026!(1\u003Eb.length)){for(var c=[],e=0;e\u003Cb.length;e++){var d=b[e].split(\".\");var f=d.shift();(\"GA1\"==f||\"1\"==f)\u0026\u00261\u003Cd.length?(f=d.shift().split(\"-\"),1==f.length\u0026\u0026(f[1]=\"1\"),f[0]*=1,f[1]*=1,d={H:f,s:d.join(\".\")}):d=void 0;d\u0026\u0026c.push(d)}if(1==c.length)return h(13),c[0].s;if(0!=c.length)return h(14),b=sa(m(a,t)).split(\".\").length,c=Ma(c,b,0),1==c.length?c[0].s:(a=La(m(a,E)),c=Ma(c,a,1),c[0]\u0026\u0026c[0].s);h(12)}},Ma=function(a,b,c){for(var e,\nd=[],f=[],g=0;g\u003Ca.length;g++){var h=a[g];h.H[c]==b?d.push(h):void 0==e||h.H[c]\u003Ce?(f=[h],e=h.H[c]):h.H[c]==e\u0026\u0026f.push(h)}return 0\u003Cd.length?d:f},sa=function(a){return 0==a.indexOf(\".\")?a.substr(1):a},Ka=function(a){return a?(1\u003Ca.length\u0026\u0026a.lastIndexOf(\"\/\")==a.length-1\u0026\u0026(a=a.substr(0,a.length-1)),0!=a.indexOf(\"\/\")\u0026\u0026(a=\"\/\"+a),a):\"\/\"},La=function(a){return a=Ka(a),\"\/\"==a?1:a.split(\"\/\").length},Jb=new RegExp(\/^https?:\\\/\\\/([^\\\/:]+)\/),Kb=\/(.*)([?\u0026#])(?:_ga=[^\u0026#]*)(?:\u0026?)(.*)\/,ba=function(a){h(48);this.target=\na;this.T=!1};ba.prototype.ca=function(a,b){if(a.tagName){if(\"a\"==a.tagName.toLowerCase())return void(a.href\u0026\u0026(a.href=ca(this,a.href,b)));if(\"form\"==a.tagName.toLowerCase())return Oa(this,a)}if(\"string\"==typeof a)return ca(this,a,b)};var ca=function(a,b,c){var e=Kb.exec(b);e\u0026\u00263\u003C=e.length\u0026\u0026(b=e[1]+(e[3]?e[2]+e[3]:\"\"));a=a.target.get(\"linkerParam\");var d=b.indexOf(\"?\");e=b.indexOf(\"#\");return c?b+=(-1==e?\"#\":\"\\x26\")+a:(c=-1==d?\"?\":\"\\x26\",b=-1==e?b+(c+a):b.substring(0,e)+c+a+b.substring(e)),b.replace(\/\u0026+_ga=\/,\n\"\\x26_ga\\x3d\")},Oa=function(a,b){if(b\u0026\u0026b.action)if(\"get\"==b.method.toLowerCase()){a=a.target.get(\"linkerParam\").split(\"\\x3d\")[1];for(var c=b.childNodes||[],e=0;e\u003Cc.length;e++)if(\"_ga\"==c[e].name)return void c[e].setAttribute(\"value\",a);c=k.createElement(\"input\");c.setAttribute(\"type\",\"hidden\");c.setAttribute(\"name\",\"_ga\");c.setAttribute(\"value\",a);b.appendChild(c)}else\"post\"==b.method.toLowerCase()\u0026\u0026(b.action=ca(a,b.action))};ba.prototype.S=function(a,b,c){function e(c){try{c=c||p.event;a:{var e=\nc.target||c.srcElement;for(c=100;e\u0026\u00260\u003Cc;){if(e.href\u0026\u0026e.nodeName.match(\/^a(?:rea)?$\/i)){var f=e;break a}e=e.parentNode;c--}f={}}(\"http:\"==f.protocol||\"https:\"==f.protocol)\u0026\u0026ta(a,f.hostname||\"\")\u0026\u0026f.href\u0026\u0026(f.href=ca(d,f.href,b))}catch(Ub){h(26)}}var d=this;this.T||(this.T=!0,T(k,\"mousedown\",e,!1),T(k,\"keyup\",e,!1));c\u0026\u0026T(k,\"submit\",function(b){if(b=b||p.event,(b=b.target||b.srcElement)\u0026\u0026b.action){var c=b.action.match(Jb);c\u0026\u0026ta(a,c[1])\u0026\u0026Oa(d,b)}})};var Pa,Mb=function(a,b,c){this.U=ub;this.aa=b;(b=c)||\n(b=(b=m(a,y))\u0026\u0026\"t0\"!=b?Lb.test(b)?\"_gat_\"+ha(m(a,A)):\"_gat_\"+ha(b):\"_gat\");this.Y=b},Qa=function(a,b,c){b.get(c)||(\"1\"==G(a.Y)[0]?b.set(c,\"\",!0):b.set(c,\"\"+S(),!0))},Lb=\/^gtm\\d+$\/,Nb=function(a){if(!a.get(\"dcLoaded\")\u0026\u0026\"cookie\"==a.get(I)){var b=a,c=b;var e=(c=c.get(Ca),\"[object Array]\"==Object.prototype.toString.call(Object(c))||(c=[]),c);e=new R(e);e.set(51);b.set(Ca,e.w);b=new Mb(a);Qa(b,a,b.U);Qa(b,a,vb);e=b;c=a;c.get(e.U)\u0026\u0026V(e.Y,\"1\",c.get(E),c.get(t),c.get(A),6E4);a.get(b.U)\u0026\u0026(a.set(Gb,1,!0),a.set(Fb,\n\"undefined\/r\/collect\",!0))}},Ob=function(){var a=p.gaGlobal=p.gaGlobal||{};return a.hid=a.hid||S()},Pb=function(a,b,c){if(!Pa){var e=k.location.hash;var d=p.name,f=\/^#?gaso=([^\u0026]*)\/;if(d=(e=(e=e\u0026\u0026e.match(f)||d\u0026\u0026d.match(f))?e[1]:G(\"GASO\")[0]||\"\")\u0026\u0026e.match(\/^(?:!([-0-9a-z.]{1,40})!)?([-.\\w]{10,1200})$\/i))V(\"GASO\",\"\"+e,c,b,a,0),window._udo||(window._udo=b),window._utcp||(window._utcp=c),a=d[1],a=\"https:\/\/www.google.com\/analytics\/web\/inpage\/pub\/inpage.js?\"+(a?\"prefix\\x3d\"+a+\"\\x26\":\"\")+S(),b=\"_gasojs\",\ne=c=void 0,a\u0026\u0026(c?(e=\"\",b\u0026\u0026va.test(b)\u0026\u0026(e=' id\\x3d\"'+b+'\"'),va.test(a)\u0026\u0026k.write(\"\\x3cscript\"+e+' src\\x3d\"'+a+'\"\\x3e\\x3c\/script\\x3e')):(c=k.createElement(\"script\"),c.type=\"text\/javascript\",c.async=!0,c.src=a,e\u0026\u0026(c.onload=e),b\u0026\u0026(c.id=b),a=k.getElementsByTagName(\"script\")[0],a.parentNode.insertBefore(c,a)));Pa=!0}},ab=\/^(UA|YT|MO|GP)-(\\d+)-(\\d+)$\/,K=function(a){function b(a,b){e.b.data.set(a,b)}function c(a,c){b(a,c);e.filters.add(a)}var e=this;this.b=new W;this.filters=new xa;b(y,a[y]);b(A,Xa(a[A]));\nb(Q,a[Q]);b(t,a[t]||U());b(E,a[E]);b(ka,a[ka]);b(aa,a[aa]);b(la,a[la]);b(ma,a[ma]);b(na,a[na]);b(Fa,a[Fa]);b(oa,a[oa]);b(Ga,a[Ga]);b(I,a[I]);b(Ea,a[Ea]);b(Da,a[Da]);b(z,a[z]);b(cb,1);b(db,\"j50\");c(wb,q);c(xb,B);c(yb,x);c(zb,L);c(Ab,Hb);c(Bb,Ib);c(Cb,Va);c(Db,F);c(Eb,Nb);Qb(this.b,a[v]);this.b.set(eb,Ob());Pb(this.b.get(A),this.b.get(t),this.b.get(E));this.ra=new qa(1E4,!0,\"gaexp10\")},Qb=function(a,b){if(\"cookie\"==m(a,I)){J=!1;var c=G(m(a,Q));if(!(c=Na(a,c))){c=m(a,t);var e=m(a,aa)||U();c=da(\"__utma\",\ne,c);void 0!=c?(h(10),c=c.O[1]+\".\"+c.O[2]):c=void 0}c\u0026\u0026(a.data.set(v,c),c=G(m(a,pa)),(c=Na(a,c))\u0026\u0026a.data.set(H,c),J=!0)}a:if(c=a.get(na),e=k.location[c?\"href\":\"search\"],c=(e=e.match(\"(?:\\x26|#|\\\\?)\"+N(\"_ga\").replace(\/([.*+?^=!:${}()|\\[\\]\\\/\\\\])\/g,\"\\\\$1\")+\"\\x3d([^\\x26#]*)\"))\u0026\u00262==e.length?e[1]:\"\")if(a.get(ma))if(-1==(e=c.indexOf(\".\")))h(22);else{var d=c.substring(0,e),f=c.substring(e+1);e=f.indexOf(\".\");c=f.substring(0,e);f=f.substring(e+1);if(\"1\"==d){if(e=f,c!=C(e,0)\u0026\u0026c!=C(e,-1)\u0026\u0026c!=C(e,-2)){h(23);\nbreak a}}else{if(\"2\"!=d){h(22);break a}if(d=f.split(\"-\",2),e=d[1],c!=C(d[0]+e,0)\u0026\u0026c!=C(d[0]+e,-1)\u0026\u0026c!=C(d[0]+e,-2)){h(53);break a}h(2);a.data.set(H,d[0])}h(11);a.data.set(v,e)}else h(21);b\u0026\u0026(h(9),a.data.set(v,N(b)));a.get(v)||((b=(b=p.gaGlobal\u0026\u0026p.gaGlobal.vid)\u0026\u0026-1!=b.search(\/^(?:utma\\.)?\\d+\\.\\d+$\/)?b:void 0)?(h(17),a.data.set(v,b)):(h(8),a.data.set(v,ua())));a.data.set(z,a.get(z)||1==ra(new qa(0,!0),void 0,M(a.get(v))));a.get(z)\u0026\u0026(b=m(a,Q),a.data.set(pa,\"_ga\"==b?\"_gid\":b+\"_gid\"));a.get(z)\u0026\u0026!a.get(H)\u0026\u0026\n(h(3),a.data.set(H,ua()));Ia(a)};K.prototype.get=function(a){return this.b.get(a)};K.prototype.set=function(a,b){this.b.set(a,b)};var Rb={pageview:[fb],event:[gb,hb,ib,jb],social:[kb,lb,mb],timing:[nb,ob,qb,pb]};K.prototype.send=function(a){if(!(1\u003Earguments.length)){var b,c;\"string\"==typeof arguments[0]?(b=arguments[0],c=[].slice.call(arguments,1)):(b=arguments[0]\u0026\u0026arguments[0][Ba],c=arguments);b\u0026\u0026(c=wa(Rb[b]||[],c),c[Ba]=b,this.b.set(c,void 0,!0),this.filters.D(this.b),this.b.data.m={},ra(this.ra,\nthis.b)\u0026\u0026this.b.get(A))}};K.prototype.ma=function(a,b){a=this;a.get(y)};var Ra=function(a){if(\"prerender\"==k.visibilityState||(a(),!1)){h(16);var b=!1,c=function(){if(!b\u0026\u0026\"prerender\"!=k.visibilityState\u0026\u0026(a(),!0)){b=!0;var e=c,d=k;d.removeEventListener?d.removeEventListener(\"visibilitychange\",e,!1):d.detachEvent\u0026\u0026d.detachEvent(\"onvisibilitychange\",e)}};T(k,\"visibilitychange\",c)}},Sb=function(a){};new D;new D;new D;var n={ga:function(){n.f=[]}};n.ga();n.D=function(a){var b=n.J.apply(n,arguments);b=\nn.f.concat(b);for(n.f=[];0\u003Cb.length\u0026\u0026!n.v(b[0])\u0026\u0026(b.shift(),!(0\u003Cn.f.length)););n.f=n.f.concat(b)};n.J=function(a){for(var b=[],c=0;c\u003Carguments.length;c++)try{var d=new Sb(arguments[c]);d.g||(d.i\u0026\u0026(d.ha=void 0),b.push(d))}catch(u){}return b};n.v=function(a){try{if(a.u)a.u.call(p,g.j(\"t0\"));else{var b=a.c==P?g:g.j(a.c);if(a.A)\"t0\"!=a.c||g.create.apply(g,a.a);else if(a.ba)g.remove(a.c);else if(b){if(a.i)return a.ha\u0026\u0026(a.ha=void 0),!0;if(a.K){var c=a.C,d=a.a,h=b.plugins_.get(a.K);h[c].apply(h,d)}else b[a.C].apply(b,\na.a)}}}catch(f){}};var g=function(a){h(1);n.D.apply(n,[arguments])};g.h={};g.P=[];g.L=0;g.answer=42;var Tb=[A,t,y];g.create=function(a){var b=wa(Tb,[].slice.call(arguments));b[y]||(b[y]=\"t0\");var c=\"\"+b[y];return g.h[c]?g.h[c]:(b=new K(b),g.h[c]=b,g.P.push(b),b)};g.remove=function(a){for(var b=0;b\u003Cg.P.length;b++)if(g.P[b].get(y)==a){g.P.splice(b,1);g.h[a]=null;break}};g.j=function(a){return g.h[a]};g.getAll=function(){return g.P.slice(0)};g.N=function(){\"ga\"!=P\u0026\u0026h(49);var a=p[P];if(!a||42!=a.answer){g.L=\na\u0026\u0026a.l;g.loaded=!0;var b=p[P]=g;r(\"create\",b,b.create);r(\"remove\",b,b.remove);r(\"getByName\",b,b.j,5);r(\"getAll\",b,b.getAll,6);b=K.prototype;r(\"get\",b,b.get,7);r(\"set\",b,b.set,4);r(\"send\",b,b.send);r(\"requireSync\",b,b.ma);b=W.prototype;r(\"get\",b,b.get);r(\"set\",b,b.set);\"https:\"==k.location.protocol||Z||!ra(new qa(1E4))||(h(36),Z=!0);(p.gaplugins=p.gaplugins||{}).Linker=ba;b=ba.prototype;r(\"decorate\",b,b.ca,20);r(\"autoLink\",b,b.S,25);a=a\u0026\u0026a.q;\"[object Array]\"==Object.prototype.toString.call(Object(a))?\nn.D.apply(g,a):h(50)}};g.da=function(){for(var a=g.getAll(),b=0;b\u003Ca.length;b++)a[b].get(y)};var Sa=g.N,Ta=p[P];Ta\u0026\u0026Ta.r?Sa():Ra(Sa);Ra(function(){n.D([\"provide\",\"render\",Ya])})}(window);var q=\"SCITYLANA\";q=q+\"_temp_\"+Math.round(2147483647*Math.random());B=slga.create(B,x,q);L=L||B.get(\"userId\")||B.get(\"clientId\");B=document.referrer?1:0;slga.remove(q);q=window;x=\"_o_r_d_e_r_sl\";var F=(new Date).getTime();q[x]=q[x]?q[x]==F?F+1:F\u003Eq[x]?F:q[x]+1:F;return q=[\"sl\\x3d1\",\"u\\x3d\"+L,\"t\\x3d\"+q[x],\"r\\x3d\"+B].join(\"\\x26\")})();"]
    },{
      "function":"__k",
      "vtp_decodeCookie":false,
      "vtp_name":"XilinxUserOrg"
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=",["escape",["macro",21],8,16],";return a?a:\"Anonymous\"})();"]
    },{
      "function":"__jsm",
      "vtp_javascript":["template","(function(){var a=",["escape",["macro",17],8,16],".closest(\"div.xdf.register\");return a.getAttribute(\"class\")})();"]
    },{
      "function":"__e"
    },{
      "function":"__v",
      "vtp_dataLayerVersion":2,
      "vtp_setDefaultValue":false,
      "vtp_name":"OptanonActiveGroups"
    },{
      "function":"__c",
      "vtp_value":"UA-11440126-15"
    },{
      "function":"__v",
      "vtp_name":"gtm.elementUrl",
      "vtp_dataLayerVersion":1
    },{
      "function":"__v",
      "vtp_name":"gtm.triggers",
      "vtp_dataLayerVersion":2,
      "vtp_setDefaultValue":true,
      "vtp_defaultValue":""
    },{
      "function":"__j",
      "vtp_name":"document.title"
    },{
      "function":"__f",
      "vtp_component":"URL"
    },{
      "function":"__u",
      "vtp_component":"PATH",
      "vtp_enableMultiQueryKeys":false,
      "vtp_enableIgnoreEmptyQueryParam":false
    },{
      "function":"__remm",
      "vtp_setDefaultValue":false,
      "vtp_input":["macro",19],
      "vtp_fullMatch":true,
      "vtp_replaceAfterMatch":true,
      "vtp_ignoreCase":true,
      "vtp_map":["list",["map","key","(.*)\/search\/site-keyword-search.html","value","EntireSite"],["map","key","(.*)\/products\/silicon-devices\/si-keyword-search.html","value","Silicon-Devices"],["map","key","(.*)\/products\/boards-and-kits\/bk-keyword-search.html","value","Boards-and-Kits"],["map","key","(.*)\/products\/intellectual-property\/ip-keyword-search.html","value","Intellectual-Property"],["map","key","(.*)\/search\/support-keyword-search.html","value","Answers_Docs_Forums"],["map","key","(.*)\/alliance\/member-keyword-search.html","value","Partners"],["map","key","(.*)\/video\/video-keyword-search.html","value","Videos"],["map","key","(.*)\/search\/press-keyword-search.html","value","Press"],["map","key","(.*)\/support\/documentation-navigation\/documentation-keyword-search.html","value","Documentation"],["map","key","(.*)\/support\/answer-navigation\/answer-keyword-search.html","value","AnswersDatabase"],["map","key","(.*)\/search\/forums-keyword-search.html","value","Forums"],["map","key","answer record,document,forums","value","Answers_Docs_Forums"],["map","key","document","value","Documentation"],["map","key","answer record","value","AnswersDatabase"],["map","key","product information","value","Partners"],["map","key","partner information","value","Partners"],["map","key","video","value","Videos"],["map","key","press release,media kit","value","Press"],["map","key","forums","value","Forums"],["map","key","site","value","EntireSite"]]
    },{
      "function":"__aev",
      "vtp_varType":"TEXT"
    },{
      "function":"__v",
      "vtp_name":"gtm.elementClasses",
      "vtp_dataLayerVersion":1
    },{
      "function":"__u",
      "vtp_enableMultiQueryKeys":false,
      "vtp_enableIgnoreEmptyQueryParam":false
    },{
      "function":"__u",
      "vtp_component":"HOST",
      "vtp_enableMultiQueryKeys":false,
      "vtp_enableIgnoreEmptyQueryParam":false
    },{
      "function":"__e"
    },{
      "function":"__v",
      "vtp_name":"gtm.elementId",
      "vtp_dataLayerVersion":1
    },{
      "function":"__v",
      "vtp_name":"gtm.elementTarget",
      "vtp_dataLayerVersion":1
    },{
      "function":"__v",
      "vtp_name":"gtm.element",
      "vtp_dataLayerVersion":1
    },{
      "function":"__v",
      "vtp_name":"gtm.elementClasses",
      "vtp_dataLayerVersion":1
    },{
      "function":"__v",
      "vtp_name":"gtm.elementId",
      "vtp_dataLayerVersion":1
    },{
      "function":"__v",
      "vtp_name":"gtm.elementTarget",
      "vtp_dataLayerVersion":1
    },{
      "function":"__v",
      "vtp_name":"gtm.elementUrl",
      "vtp_dataLayerVersion":1
    },{
      "function":"__aev",
      "vtp_varType":"TEXT"
    }],
  "tags":[{
      "function":"__ua",
      "priority":3,
      "once_per_event":true,
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_useDebugVersion":false,
      "vtp_useHashAutoLink":false,
      "vtp_trackType":"TRACK_PAGEVIEW",
      "vtp_decorateFormsAutoLink":false,
      "vtp_enableLinkId":true,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]],["map","index","2","dimension",["macro",22]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "tag_id":1
    },{
      "function":"__cegg",
      "priority":2,
      "once_per_event":true,
      "vtp_usersNumericId":"00854198",
      "tag_id":36
    },{
      "function":"__mf",
      "priority":1,
      "once_per_event":true,
      "vtp_projectId":"a3971bc5-37ea-43dc-b133-29bc1d15e686",
      "tag_id":39
    },{
      "function":"__html",
      "priority":1,
      "once_per_event":true,
      "vtp_html":"\u003Cscript type=\"text\/gtmscript\"\u003Evar mouseflowPath=document.location.host+document.location.pathname+document.location.hash;document.addEventListener(\"hashchange\",function(){window._mfq=window._mfq||[];window._mfq.push([\"newPageView\",document.location.host+document.location.pathname+document.location.hash])});\u003C\/script\u003E",
      "vtp_supportDocumentWrite":false,
      "vtp_enableIframeMode":false,
      "vtp_enableEditJsMacroBehavior":false,
      "tag_id":40
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"External-Link",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":["macro",27],
      "vtp_eventLabel":["macro",2],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":2
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"Mailto",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":["macro",27],
      "vtp_eventLabel":["macro",2],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":5
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"Downloads",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":"Click",
      "vtp_eventLabel":["macro",27],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_fieldsToSet":["list",["map","fieldName","XilinxUserOrg","value",["macro",21]]],
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]],["map","index","1","dimension",["macro",21]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":6
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"404 Error",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":["macro",2],
      "vtp_eventLabel":["macro",30],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":9
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"Answer-Feedback",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":["macro",14],
      "vtp_eventLabel":["macro",3],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":10
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_useDebugVersion":false,
      "vtp_fieldsToSet":["list",["map","fieldName","page","value",["macro",4]],["map","fieldName","title","value",["macro",5]]],
      "vtp_useHashAutoLink":false,
      "vtp_trackType":"TRACK_PAGEVIEW",
      "vtp_decorateFormsAutoLink":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "tag_id":11
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"Tab",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":"Open",
      "vtp_eventLabel":["template",["macro",2],"#",["macro",7]],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":12
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"Tab",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":"Open",
      "vtp_eventLabel":["template",["macro",2],"#",["macro",6]],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":16
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_useDebugVersion":false,
      "vtp_fieldsToSet":["list",["map","fieldName","page","value",["template",["macro",31],"#",["macro",7]]]],
      "vtp_useHashAutoLink":false,
      "vtp_trackType":"TRACK_PAGEVIEW",
      "vtp_decorateFormsAutoLink":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "tag_id":17
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_useDebugVersion":false,
      "vtp_fieldsToSet":["list",["map","fieldName","page","value",["template",["macro",31],"#",["macro",6]]]],
      "vtp_useHashAutoLink":false,
      "vtp_trackType":"TRACK_PAGEVIEW",
      "vtp_decorateFormsAutoLink":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "tag_id":18
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_useDebugVersion":false,
      "vtp_fieldsToSet":["list",["map","fieldName","page","value",["template",["macro",31],"#",["macro",8]]]],
      "vtp_useHashAutoLink":false,
      "vtp_trackType":"TRACK_PAGEVIEW",
      "vtp_decorateFormsAutoLink":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "tag_id":19
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"Search",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":["macro",13],
      "vtp_eventLabel":["template","Search Option=",["macro",32]," | q=",["macro",9]," | ",["macro",2]],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":20
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"Search",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":["macro",13],
      "vtp_eventLabel":["template","Search Option=",["macro",32]," | q=",["macro",9]," | ",["macro",2]],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":23
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_useDebugVersion":false,
      "vtp_eventCategory":"Share",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":"Click",
      "vtp_eventLabel":["template",["macro",15]," | ",["macro",2]],
      "vtp_overrideGaSettings":true,
      "vtp_doubleClick":false,
      "vtp_setTrackerName":false,
      "vtp_enableLinkId":false,
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_enableEcommerce":false,
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":25
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_overrideGaSettings":true,
      "vtp_eventCategory":"Video",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":"Click",
      "vtp_eventLabel":["template",["macro",2]," | ",["macro",18]],
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":26
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_overrideGaSettings":true,
      "vtp_eventCategory":["template","BK ",["macro",33]],
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":"Click",
      "vtp_eventLabel":["template",["macro",2]," | ",["macro",33]],
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":27
    },{
      "function":"__ua",
      "once_per_event":true,
      "vtp_nonInteraction":false,
      "vtp_overrideGaSettings":true,
      "vtp_eventCategory":"Community Hackter",
      "vtp_trackType":"TRACK_EVENT",
      "vtp_eventAction":"Click",
      "vtp_eventLabel":["macro",27],
      "vtp_dimension":["list",["map","index","3","dimension",["macro",20]]],
      "vtp_trackingId":["macro",26],
      "vtp_enableRecaptchaOption":false,
      "vtp_enableUaRlsa":false,
      "vtp_enableUseInternalVersion":false,
      "vtp_enableFirebaseCampaignData":true,
      "vtp_trackTypeIsEvent":true,
      "tag_id":35
    },{
      "function":"__awct",
      "metadata":["map"],
      "once_per_event":true,
      "vtp_enableConversionLinker":true,
      "vtp_conversionCookiePrefix":"_gcl",
      "vtp_conversionId":"1015603422",
      "vtp_conversionLabel":"ECKiCJPkr6sBEN7Bo-QD",
      "vtp_url":["macro",35],
      "vtp_enableProductReportingCheckbox":true,
      "vtp_enableNewCustomerReportingCheckbox":false,
      "vtp_enableEnhancedConversionsCheckbox":false,
      "vtp_enableRdpCheckbox":false,
      "tag_id":43
    },{
      "function":"__gclidw",
      "metadata":["map"],
      "once_per_event":true,
      "vtp_enableCrossDomain":true,
      "vtp_acceptIncoming":true,
      "vtp_linkerDomains":"cvent.com",
      "vtp_formDecoration":false,
      "vtp_urlPosition":"query",
      "vtp_enableCookieOverrides":false,
      "vtp_enableCrossDomainFeature":true,
      "vtp_enableCookieUpdateFeature":false,
      "tag_id":45
    },{
      "function":"__lcl",
      "vtp_waitForTags":true,
      "vtp_checkValidation":true,
      "vtp_waitForTagsTimeout":"2000",
      "vtp_uniqueTriggerId":"2077980_4",
      "tag_id":46
    },{
      "function":"__lcl",
      "vtp_waitForTags":true,
      "vtp_checkValidation":true,
      "vtp_waitForTagsTimeout":"2000",
      "vtp_uniqueTriggerId":"2077980_7",
      "tag_id":47
    },{
      "function":"__lcl",
      "vtp_waitForTags":true,
      "vtp_checkValidation":true,
      "vtp_waitForTagsTimeout":"2000",
      "vtp_uniqueTriggerId":"2077980_8",
      "tag_id":48
    },{
      "function":"__lcl",
      "vtp_waitForTags":true,
      "vtp_checkValidation":true,
      "vtp_waitForTagsTimeout":"2000",
      "vtp_uniqueTriggerId":"2077980_9",
      "tag_id":49
    },{
      "function":"__lcl",
      "vtp_waitForTags":true,
      "vtp_checkValidation":true,
      "vtp_waitForTagsTimeout":"2000",
      "vtp_uniqueTriggerId":"2077980_10",
      "tag_id":50
    },{
      "function":"__cl",
      "tag_id":51
    },{
      "function":"__lcl",
      "vtp_waitForTags":true,
      "vtp_checkValidation":true,
      "vtp_waitForTagsTimeout":"2000",
      "vtp_uniqueTriggerId":"2077980_13",
      "tag_id":52
    },{
      "function":"__cl",
      "tag_id":53
    },{
      "function":"__cl",
      "tag_id":54
    },{
      "function":"__fsl",
      "vtp_checkValidation":false,
      "vtp_waitForTagsTimeout":"2000",
      "vtp_uniqueTriggerId":"2077980_21",
      "tag_id":55
    },{
      "function":"__cl",
      "tag_id":56
    },{
      "function":"__cl",
      "tag_id":57
    },{
      "function":"__cl",
      "tag_id":58
    },{
      "function":"__lcl",
      "vtp_waitForTags":false,
      "vtp_checkValidation":false,
      "vtp_waitForTagsTimeout":"2000",
      "vtp_uniqueTriggerId":"2077980_57",
      "tag_id":59
    },{
      "function":"__cl",
      "tag_id":60
    },{
      "function":"__html",
      "once_per_event":true,
      "vtp_html":"\u003Cscript type=\"text\/gtmscript\"\u003E(function(){var c=$(\".search-container .CoveoSearchInterface\"),d=$(\".mobile-search-container .CoveoSearchInterface\"),e=$(\".xilinxSearchBox .CoveoSearchInterface\");c.on(\"beforeRedirect\",function(a,b){window.dataLayer.push({event:\"CoveoFormSubmit\",enterElement:a.target})});d.on(\"beforeRedirect\",function(a,b){window.dataLayer.push({event:\"CoveoFormSubmit\",enterElement:a.target})});e.on(\"beforeRedirect\",function(a,b){window.dataLayer.push({event:\"CoveoFormSubmit\",enterElement:a.target})})})();\u003C\/script\u003E",
      "vtp_supportDocumentWrite":false,
      "vtp_enableIframeMode":false,
      "vtp_enableEditJsMacroBehavior":false,
      "tag_id":34
    },{
      "function":"__html",
      "metadata":["map"],
      "once_per_event":true,
      "vtp_html":"\u003Cscript type=\"text\/gtmscript\"\u003Ega(\"require\",\"linker\");ga(\"link:autoLink\",[\"cvent.com\"],!1,!0);\u003C\/script\u003E",
      "vtp_supportDocumentWrite":false,
      "vtp_enableIframeMode":false,
      "vtp_enableEditJsMacroBehavior":false,
      "tag_id":44
    }],
  "predicates":[{
      "function":"_eq",
      "arg0":["macro",24],
      "arg1":"gtm.js"
    },{
      "function":"_eq",
      "arg0":["macro",8],
      "arg1":"undefined"
    },{
      "function":"_re",
      "arg0":["macro",25],
      "arg1":",2,"
    },{
      "function":"_cn",
      "arg0":["macro",27],
      "arg1":"xilinx.com"
    },{
      "function":"_cn",
      "arg0":["macro",27],
      "arg1":"http"
    },{
      "function":"_eq",
      "arg0":["macro",24],
      "arg1":"gtm.linkClick"
    },{
      "function":"_re",
      "arg0":["macro",28],
      "arg1":"(^$|((^|,)2077980_4($|,)))"
    },{
      "function":"_sw",
      "arg0":["macro",27],
      "arg1":"mailto"
    },{
      "function":"_re",
      "arg0":["macro",28],
      "arg1":"(^$|((^|,)2077980_7($|,)))"
    },{
      "function":"_re",
      "arg0":["macro",27],
      "arg1":"\\.(c|cpp|dat|doc|docx|exe|gif|gz|jpg|log|mp3|pdf|png|ppt|pptx|rar|svg|tar|tgz|txt|xls|xlsm|xlsx|zip|asc|sig)$",
      "ignore_case":true
    },{
      "function":"_re",
      "arg0":["macro",28],
      "arg1":"(^$|((^|,)2077980_8($|,)))"
    },{
      "function":"_cn",
      "arg0":["macro",29],
      "arg1":"Page Not Found"
    },{
      "function":"_eq",
      "arg0":["macro",24],
      "arg1":"gtm.dom"
    },{
      "function":"_css",
      "arg0":["macro",17],
      "arg1":"form button.btn.btn-info"
    },{
      "function":"_cn",
      "arg0":["macro",2],
      "arg1":"support\/answers\/"
    },{
      "function":"_eq",
      "arg0":["macro",24],
      "arg1":"gtm.click"
    },{
      "function":"_re",
      "arg0":["macro",27],
      "arg1":"\\.(pdf)$",
      "ignore_case":true
    },{
      "function":"_re",
      "arg0":["macro",28],
      "arg1":"(^$|((^|,)2077980_13($|,)))"
    },{
      "function":"_cn",
      "arg0":["macro",17],
      "arg1":"#tabAnchor-"
    },{
      "function":"_cn",
      "arg0":["macro",12],
      "arg1":"browse-answer-record-anchor"
    },{
      "function":"_cn",
      "arg0":["macro",1],
      "arg1":"#tabAnchor-"
    },{
      "function":"_eq",
      "arg0":["macro",16],
      "arg1":"Not-Support-Partner"
    },{
      "function":"_re",
      "arg0":["macro",9],
      "arg1":".+"
    },{
      "function":"_eq",
      "arg0":["macro",24],
      "arg1":"CoveoFormSubmit"
    },{
      "function":"_eq",
      "arg0":["macro",16],
      "arg1":"Xilinx-Support-Partner"
    },{
      "function":"_cn",
      "arg0":["macro",2],
      "arg1":"xilinx.com"
    },{
      "function":"_re",
      "arg0":["macro",15],
      "arg1":"^((?!none).)*$",
      "ignore_case":true
    },{
      "function":"_re",
      "arg0":["macro",15],
      "arg1":"^((?!undefined).)*$",
      "ignore_case":true
    },{
      "function":"_re",
      "arg0":["macro",18],
      "arg1":"video-(info|links|documents)+"
    },{
      "function":"_re",
      "arg0":["macro",33],
      "arg1":"^(Show Documentation|Contact Sales)$"
    },{
      "function":"_sw",
      "arg0":["macro",31],
      "arg1":"\/products\/boards-and-kits"
    },{
      "function":"_sw",
      "arg0":["macro",27],
      "arg1":"https:\/\/www.hackster.io"
    },{
      "function":"_eq",
      "arg0":["macro",34],
      "arg1":"disableHover post tutorial"
    },{
      "function":"_re",
      "arg0":["macro",28],
      "arg1":"(^$|((^|,)2077980_57($|,)))"
    },{
      "function":"_ew",
      "arg0":["macro",2],
      "arg1":"\/products\/design-tools\/developer-forum.html"
    },{
      "function":"_re",
      "arg0":["macro",2],
      "arg1":".*"
    },{
      "function":"_re",
      "arg0":["macro",2],
      "arg1":".*",
      "ignore_case":true
    }],
  "rules":[
    [["if",0],["add",0,1,2,3,28,30,31,32,33,34,35,36,37,38]],
    [["if",4,5,6],["unless",3],["add",4]],
    [["if",5,7,8],["add",5]],
    [["if",5,9,10],["add",6]],
    [["if",11,12],["add",7]],
    [["if",13,14,15],["add",8]],
    [["if",5,16,17],["add",9]],
    [["if",15,18],["unless",19],["add",10,12]],
    [["if",15,20],["add",11,13]],
    [["if",0],["unless",1],["add",14],["block",0]],
    [["if",21,22,23],["add",15]],
    [["if",22,23,24],["add",16]],
    [["if",15,25,26,27],["add",17]],
    [["if",15,28],["add",18]],
    [["if",15,29,30],["add",19]],
    [["if",5,31,32,33],["add",20]],
    [["if",0,34],["add",21,22,39]],
    [["if",0,35],["add",23,26,27,29]],
    [["if",0,36],["add",24,25]],
    [["if",0],["unless",2],["block",0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,21,22,38,3,39]]]
},
"runtime":[]




};
/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
var aa,ba="function"==typeof Object.create?Object.create:function(a){var b=function(){};b.prototype=a;return new b},ca;if("function"==typeof Object.setPrototypeOf)ca=Object.setPrototypeOf;else{var ha;a:{var ia={pf:!0},ja={};try{ja.__proto__=ia;ha=ja.pf;break a}catch(a){}ha=!1}ca=ha?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var ka=ca,la=this||self,na=/^[\w+/_-]+[=]{0,2}$/,oa=null;var qa=function(){},ra=function(a){return"function"==typeof a},g=function(a){return"string"==typeof a},sa=function(a){return"number"==typeof a&&!isNaN(a)},va=function(a){return"[object Array]"==Object.prototype.toString.call(Object(a))},q=function(a,b){if(Array.prototype.indexOf){var c=a.indexOf(b);return"number"==typeof c?c:-1}for(var d=0;d<a.length;d++)if(a[d]===b)return d;return-1},wa=function(a,b){if(a&&va(a))for(var c=0;c<a.length;c++)if(a[c]&&b(a[c]))return a[c]},xa=function(a,b){if(!sa(a)||
!sa(b)||a>b)a=0,b=2147483647;return Math.floor(Math.random()*(b-a+1)+a)},za=function(a,b){for(var c=new ya,d=0;d<a.length;d++)c.set(a[d],!0);for(var e=0;e<b.length;e++)if(c.get(b[e]))return!0;return!1},Aa=function(a,b){for(var c in a)Object.prototype.hasOwnProperty.call(a,c)&&b(c,a[c])},Ba=function(a){return Math.round(Number(a))||0},Ca=function(a){return"false"==String(a).toLowerCase()?!1:!!a},Ea=function(a){var b=[];if(va(a))for(var c=0;c<a.length;c++)b.push(String(a[c]));return b},Fa=function(a){return a?
a.replace(/^\s+|\s+$/g,""):""},Ga=function(){return(new Date).getTime()},ya=function(){this.prefix="gtm.";this.values={}};ya.prototype.set=function(a,b){this.values[this.prefix+a]=b};ya.prototype.get=function(a){return this.values[this.prefix+a]};
var Ha=function(a,b,c){return a&&a.hasOwnProperty(b)?a[b]:c},Ia=function(a){var b=!1;return function(){if(!b)try{a()}catch(c){}b=!0}},Ja=function(a,b){for(var c in b)b.hasOwnProperty(c)&&(a[c]=b[c])},Ka=function(a){for(var b in a)if(a.hasOwnProperty(b))return!0;return!1},La=function(a,b){for(var c=[],d=0;d<a.length;d++)c.push(a[d]),c.push.apply(c,b[a[d]]||[]);return c},Ma=function(a,b){for(var c={},d=c,e=a.split("."),f=0;f<e.length-1;f++)d=d[e[f]]={};d[e[e.length-1]]=b;return c},Na=function(a){var b=
[];Aa(a,function(c,d){10>c.length&&d&&b.push(c)});return b.join(",")};/*
 jQuery v1.9.1 (c) 2005, 2012 jQuery Foundation, Inc. jquery.org/license. */
var Oa=/\[object (Boolean|Number|String|Function|Array|Date|RegExp)\]/,Pa=function(a){if(null==a)return String(a);var b=Oa.exec(Object.prototype.toString.call(Object(a)));return b?b[1].toLowerCase():"object"},Qa=function(a,b){return Object.prototype.hasOwnProperty.call(Object(a),b)},Ra=function(a){if(!a||"object"!=Pa(a)||a.nodeType||a==a.window)return!1;try{if(a.constructor&&!Qa(a,"constructor")&&!Qa(a.constructor.prototype,"isPrototypeOf"))return!1}catch(c){return!1}for(var b in a);return void 0===
b||Qa(a,b)},C=function(a,b){var c=b||("array"==Pa(a)?[]:{}),d;for(d in a)if(Qa(a,d)){var e=a[d];"array"==Pa(e)?("array"!=Pa(c[d])&&(c[d]=[]),c[d]=C(e,c[d])):Ra(e)?(Ra(c[d])||(c[d]={}),c[d]=C(e,c[d])):c[d]=e}return c};
var Sa=[],Ta={"\x00":"&#0;",'"':"&quot;","&":"&amp;","'":"&#39;","<":"&lt;",">":"&gt;","\t":"&#9;","\n":"&#10;","\x0B":"&#11;","\f":"&#12;","\r":"&#13;"," ":"&#32;","-":"&#45;","/":"&#47;","=":"&#61;","`":"&#96;","\u0085":"&#133;","\u00a0":"&#160;","\u2028":"&#8232;","\u2029":"&#8233;"},Ua=function(a){return Ta[a]},Va=/[\x00\x22\x26\x27\x3c\x3e]/g;var bb=/[\x00\x08-\x0d\x22\x26\x27\/\x3c-\x3e\\\x85\u2028\u2029]/g,cb={"\x00":"\\x00","\b":"\\x08","\t":"\\t","\n":"\\n","\x0B":"\\x0b",
"\f":"\\f","\r":"\\r",'"':"\\x22","&":"\\x26","'":"\\x27","/":"\\/","<":"\\x3c","=":"\\x3d",">":"\\x3e","\\":"\\\\","\u0085":"\\x85","\u2028":"\\u2028","\u2029":"\\u2029",$:"\\x24","(":"\\x28",")":"\\x29","*":"\\x2a","+":"\\x2b",",":"\\x2c","-":"\\x2d",".":"\\x2e",":":"\\x3a","?":"\\x3f","[":"\\x5b","]":"\\x5d","^":"\\x5e","{":"\\x7b","|":"\\x7c","}":"\\x7d"},db=function(a){return cb[a]};
Sa[8]=function(a){if(null==a)return" null ";switch(typeof a){case "boolean":case "number":return" "+a+" ";default:return"'"+String(String(a)).replace(bb,db)+"'"}};var mb=/[\x00- \x22\x27-\x29\x3c\x3e\\\x7b\x7d\x7f\x85\xa0\u2028\u2029\uff01\uff03\uff04\uff06-\uff0c\uff0f\uff1a\uff1b\uff1d\uff1f\uff20\uff3b\uff3d]/g,nb={"\x00":"%00","\u0001":"%01","\u0002":"%02","\u0003":"%03","\u0004":"%04","\u0005":"%05","\u0006":"%06","\u0007":"%07","\b":"%08","\t":"%09","\n":"%0A","\x0B":"%0B","\f":"%0C","\r":"%0D","\u000e":"%0E","\u000f":"%0F","\u0010":"%10",
"\u0011":"%11","\u0012":"%12","\u0013":"%13","\u0014":"%14","\u0015":"%15","\u0016":"%16","\u0017":"%17","\u0018":"%18","\u0019":"%19","\u001a":"%1A","\u001b":"%1B","\u001c":"%1C","\u001d":"%1D","\u001e":"%1E","\u001f":"%1F"," ":"%20",'"':"%22","'":"%27","(":"%28",")":"%29","<":"%3C",">":"%3E","\\":"%5C","{":"%7B","}":"%7D","\u007f":"%7F","\u0085":"%C2%85","\u00a0":"%C2%A0","\u2028":"%E2%80%A8","\u2029":"%E2%80%A9","\uff01":"%EF%BC%81","\uff03":"%EF%BC%83","\uff04":"%EF%BC%84","\uff06":"%EF%BC%86",
"\uff07":"%EF%BC%87","\uff08":"%EF%BC%88","\uff09":"%EF%BC%89","\uff0a":"%EF%BC%8A","\uff0b":"%EF%BC%8B","\uff0c":"%EF%BC%8C","\uff0f":"%EF%BC%8F","\uff1a":"%EF%BC%9A","\uff1b":"%EF%BC%9B","\uff1d":"%EF%BC%9D","\uff1f":"%EF%BC%9F","\uff20":"%EF%BC%A0","\uff3b":"%EF%BC%BB","\uff3d":"%EF%BC%BD"},ob=function(a){return nb[a]};Sa[16]=function(a){return a};var qb;
var rb=[],sb=[],tb=[],ub=[],wb=[],xb={},yb,zb,Ab,Bb=function(a,b){var c={};c["function"]="__"+a;for(var d in b)b.hasOwnProperty(d)&&(c["vtp_"+d]=b[d]);return c},Cb=function(a,b){var c=a["function"];if(!c)throw Error("Error: No function name given for function call.");var d=xb[c],e={},f;for(f in a)a.hasOwnProperty(f)&&0===f.indexOf("vtp_")&&(e[void 0!==d?f:f.substr(4)]=a[f]);return void 0!==d?d(e):qb(c,e,b)},Eb=function(a,b,c){c=c||[];var d={},e;for(e in a)a.hasOwnProperty(e)&&(d[e]=Db(a[e],b,c));
return d},Fb=function(a){var b=a["function"];if(!b)throw"Error: No function name given for function call.";var c=xb[b];return c?c.priorityOverride||0:0},Db=function(a,b,c){if(va(a)){var d;switch(a[0]){case "function_id":return a[1];case "list":d=[];for(var e=1;e<a.length;e++)d.push(Db(a[e],b,c));return d;case "macro":var f=a[1];if(c[f])return;var h=rb[f];if(!h||b.Lc(h))return;c[f]=!0;try{var k=Eb(h,b,c);k.vtp_gtmEventId=b.id;d=Cb(k,b);Ab&&(d=Ab.Of(d,k))}catch(y){b.ue&&b.ue(y,Number(f)),d=!1}c[f]=
!1;return d;case "map":d={};for(var l=1;l<a.length;l+=2)d[Db(a[l],b,c)]=Db(a[l+1],b,c);return d;case "template":d=[];for(var m=!1,n=1;n<a.length;n++){var r=Db(a[n],b,c);zb&&(m=m||r===zb.ub);d.push(r)}return zb&&m?zb.Rf(d):d.join("");case "escape":d=Db(a[1],b,c);if(zb&&va(a[1])&&"macro"===a[1][0]&&zb.og(a))return zb.Kg(d);d=String(d);for(var u=2;u<a.length;u++)Sa[a[u]]&&(d=Sa[a[u]](d));return d;case "tag":var p=a[1];if(!ub[p])throw Error("Unable to resolve tag reference "+p+".");return d={he:a[2],
index:p};case "zb":var t={arg0:a[2],arg1:a[3],ignore_case:a[5]};t["function"]=a[1];var v=Gb(t,b,c),w=!!a[4];return w||2!==v?w!==(1===v):null;default:throw Error("Attempting to expand unknown Value type: "+a[0]+".");}}return a},Gb=function(a,b,c){try{return yb(Eb(a,b,c))}catch(d){JSON.stringify(a)}return 2};var Ib=function(){var a=function(b){return{toString:function(){return b}}};return{rd:a("convert_case_to"),sd:a("convert_false_to"),td:a("convert_null_to"),ud:a("convert_true_to"),vd:a("convert_undefined_to"),sh:a("debug_mode_metadata"),ra:a("function"),Re:a("instance_name"),Ve:a("live_only"),Xe:a("malware_disabled"),Ye:a("metadata"),th:a("original_vendor_template_id"),bf:a("once_per_event"),Ed:a("once_per_load"),Md:a("setup_tags"),Od:a("tag_id"),Pd:a("teardown_tags")}}();var Jb=null,Mb=function(a){function b(r){for(var u=0;u<r.length;u++)d[r[u]]=!0}var c=[],d=[];Jb=Kb(a);for(var e=0;e<sb.length;e++){var f=sb[e],h=Lb(f);if(h){for(var k=f.add||[],l=0;l<k.length;l++)c[k[l]]=!0;b(f.block||[])}else null===h&&b(f.block||[])}for(var m=[],n=0;n<ub.length;n++)c[n]&&!d[n]&&(m[n]=!0);return m},Lb=function(a){for(var b=a["if"]||[],c=0;c<b.length;c++){var d=Jb(b[c]);if(0===d)return!1;if(2===d)return null}for(var e=a.unless||[],f=0;f<e.length;f++){var h=Jb(e[f]);if(2===h)return null;
if(1===h)return!1}return!0},Kb=function(a){var b=[];return function(c){void 0===b[c]&&(b[c]=Gb(tb[c],a));return b[c]}};/*
 Copyright (c) 2014 Derek Brans, MIT license https://github.com/krux/postscribe/blob/master/LICENSE. Portions derived from simplehtmlparser, which is licensed under the Apache License, Version 2.0 */
var D=window,E=document,gc=navigator,hc=E.currentScript&&E.currentScript.src,ic=function(a,b){var c=D[a];D[a]=void 0===c?b:c;return D[a]},jc=function(a,b){b&&(a.addEventListener?a.onload=b:a.onreadystatechange=function(){a.readyState in{loaded:1,complete:1}&&(a.onreadystatechange=null,b())})},kc=function(a,b,c){var d=E.createElement("script");d.type="text/javascript";d.async=!0;d.src=a;jc(d,b);c&&(d.onerror=c);var e;if(null===oa)b:{var f=la.document,h=f.querySelector&&f.querySelector("script[nonce]");
if(h){var k=h.nonce||h.getAttribute("nonce");if(k&&na.test(k)){oa=k;break b}}oa=""}e=oa;e&&d.setAttribute("nonce",e);var l=E.getElementsByTagName("script")[0]||E.body||E.head;l.parentNode.insertBefore(d,l);return d},lc=function(){if(hc){var a=hc.toLowerCase();if(0===a.indexOf("https://"))return 2;if(0===a.indexOf("http://"))return 3}return 1},mc=function(a,b){var c=E.createElement("iframe");c.height="0";c.width="0";c.style.display="none";c.style.visibility="hidden";var d=E.body&&E.body.lastChild||
E.body||E.head;d.parentNode.insertBefore(c,d);jc(c,b);void 0!==a&&(c.src=a);return c},nc=function(a,b,c){var d=new Image(1,1);d.onload=function(){d.onload=null;b&&b()};d.onerror=function(){d.onerror=null;c&&c()};d.src=a;return d},oc=function(a,b,c,d){a.addEventListener?a.addEventListener(b,c,!!d):a.attachEvent&&a.attachEvent("on"+b,c)},pc=function(a,b,c){a.removeEventListener?a.removeEventListener(b,c,!1):a.detachEvent&&a.detachEvent("on"+b,c)},G=function(a){D.setTimeout(a,0)},qc=function(a,b){return a&&
b&&a.attributes&&a.attributes[b]?a.attributes[b].value:null},sc=function(a){var b=a.innerText||a.textContent||"";b&&" "!=b&&(b=b.replace(/^[\s\xa0]+|[\s\xa0]+$/g,""));b&&(b=b.replace(/(\xa0+|\s{2,}|\n|\r\t)/g," "));return b},tc=function(a){var b=E.createElement("div");b.innerHTML="A<div>"+a+"</div>";b=b.lastChild;for(var c=[];b.firstChild;)c.push(b.removeChild(b.firstChild));return c},uc=function(a,b,c){c=c||100;for(var d={},e=0;e<b.length;e++)d[b[e]]=!0;for(var f=a,h=0;f&&h<=c;h++){if(d[String(f.tagName).toLowerCase()])return f;
f=f.parentElement}return null},vc=function(a){gc.sendBeacon&&gc.sendBeacon(a)||nc(a)},wc=function(a,b){var c=a[b];c&&"string"===typeof c.animVal&&(c=c.animVal);return c};var yc=function(a){return xc?E.querySelectorAll(a):null},zc=function(a,b){if(!xc)return null;if(Element.prototype.closest)try{return a.closest(b)}catch(e){return null}var c=Element.prototype.matches||Element.prototype.webkitMatchesSelector||Element.prototype.mozMatchesSelector||Element.prototype.msMatchesSelector||Element.prototype.oMatchesSelector,d=a;if(!E.documentElement.contains(d))return null;do{try{if(c.call(d,b))return d}catch(e){break}d=d.parentElement||d.parentNode}while(null!==d&&1===d.nodeType);
return null},Ac=!1;if(E.querySelectorAll)try{var Bc=E.querySelectorAll(":root");Bc&&1==Bc.length&&Bc[0]==E.documentElement&&(Ac=!0)}catch(a){}var xc=Ac;var H={qa:"_ee",nc:"event_callback",tb:"event_timeout",D:"gtag.config",X:"allow_ad_personalization_signals",oc:"restricted_data_processing",Qa:"allow_google_signals",Y:"cookie_expires",sb:"cookie_update",Ra:"session_duration",ba:"user_properties"};H.fe=[H.X,H.Qa,H.sb];H.ne=[H.Y,H.tb,H.Ra];var Rc=/[A-Z]+/,Sc=/\s/,Tc=function(a){if(g(a)&&(a=Fa(a),!Sc.test(a))){var b=a.indexOf("-");if(!(0>b)){var c=a.substring(0,b);if(Rc.test(c)){for(var d=a.substring(b+1).split("/"),e=0;e<d.length;e++)if(!d[e])return;return{id:a,prefix:c,containerId:c+"-"+d[0],o:d}}}}},Vc=function(a){for(var b={},c=0;c<a.length;++c){var d=Tc(a[c]);d&&(b[d.id]=d)}Uc(b);var e=[];Aa(b,function(f,h){e.push(h)});return e};
function Uc(a){var b=[],c;for(c in a)if(a.hasOwnProperty(c)){var d=a[c];"AW"===d.prefix&&d.o[1]&&b.push(d.containerId)}for(var e=0;e<b.length;++e)delete a[b[e]]};var Wc={},I=null,Xc=Math.random();Wc.s="GTM-5RHQV7";Wc.yb="2q2";var Yc={__cl:!0,__ecl:!0,__ehl:!0,__evl:!0,__fal:!0,__fil:!0,__fsl:!0,__hl:!0,__jel:!0,__lcl:!0,__sdl:!0,__tl:!0,__ytl:!0,__paused:!0,__tg:!0},Zc="www.googletagmanager.com/gtm.js";var $c=Zc,bd=null,cd=null,dd=null,ed="//www.googletagmanager.com/a?id="+Wc.s+"&cv=46",fd={},gd={},hd=function(){var a=I.sequence||0;I.sequence=a+1;return a};var id={},P=function(a,b){id[a]=id[a]||[];id[a][b]=!0},jd=function(a){for(var b=[],c=id[a]||[],d=0;d<c.length;d++)c[d]&&(b[Math.floor(d/6)]^=1<<d%6);for(var e=0;e<b.length;e++)b[e]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(b[e]||0);return b.join("")};
var kd=function(){return"&tc="+ub.filter(function(a){return a}).length},nd=function(){ld||(ld=D.setTimeout(md,500))},md=function(){ld&&(D.clearTimeout(ld),ld=void 0);void 0===od||pd[od]&&!qd&&!rd||(sd[od]||td.qg()||0>=ud--?(P("GTM",1),sd[od]=!0):(td.Tg(),nc(vd()),pd[od]=!0,wd=xd=rd=qd=""))},vd=function(){var a=od;if(void 0===a)return"";var b=jd("GTM"),c=jd("TAGGING");return[yd,pd[a]?"":"&es=1",zd[a],b?"&u="+b:"",c?"&ut="+c:"",kd(),qd,rd,xd,wd,"&z=0"].join("")},Ad=function(){return[ed,"&v=3&t=t","&pid="+
xa(),"&rv="+Wc.yb].join("")},Bd="0.005000">Math.random(),yd=Ad(),Cd=function(){yd=Ad()},pd={},qd="",rd="",wd="",xd="",od=void 0,zd={},sd={},ld=void 0,td=function(a,b){var c=0,d=0;return{qg:function(){if(c<a)return!1;Ga()-d>=b&&(c=0);return c>=a},Tg:function(){Ga()-d>=b&&(c=0);c++;d=Ga()}}}(2,1E3),ud=1E3,Dd=function(a,b){if(Bd&&!sd[a]&&od!==a){md();od=a;wd=qd="";var c;c=0===b.indexOf("gtm.")?encodeURIComponent(b):"*";zd[a]="&e="+c+"&eid="+a;nd()}},Ed=function(a,b,c){if(Bd&&!sd[a]&&
b){a!==od&&(md(),od=a);var d,e=String(b[Ib.ra]||"").replace(/_/g,"");0===e.indexOf("cvt")&&(e="cvt");d=e;var f=c+d;qd=qd?qd+"."+f:"&tr="+f;var h=b["function"];if(!h)throw Error("Error: No function name given for function call.");var k=(xb[h]?"1":"2")+d;wd=wd?wd+"."+k:"&ti="+k;nd();2022<=vd().length&&md()}},Fd=function(a,b,c){if(Bd&&!sd[a]){a!==od&&(md(),od=a);var d=c+b;rd=rd?rd+
"."+d:"&epr="+d;nd();2022<=vd().length&&md()}};var Gd={},Hd=new ya,Id={},Jd={},Md={name:"dataLayer",set:function(a,b){C(Ma(a,b),Id);Kd()},get:function(a){return Ld(a,2)},reset:function(){Hd=new ya;Id={};Kd()}},Ld=function(a,b){if(2!=b){var c=Hd.get(a);if(Bd){var d=Nd(a);c!==d&&P("GTM",5)}return c}return Nd(a)},Nd=function(a,b,c){var d=a.split("."),e=!1,f=void 0;return e?f:Pd(d)},Pd=function(a){for(var b=Id,c=0;c<a.length;c++){if(null===b)return!1;if(void 0===b)break;b=b[a[c]]}return b};
var Rd=function(a,b){Jd.hasOwnProperty(a)||(Hd.set(a,b),C(Ma(a,b),Id),Kd())},Kd=function(a){Aa(Jd,function(b,c){Hd.set(b,c);C(Ma(b,void 0),Id);C(Ma(b,c),Id);a&&delete Jd[b]})},Sd=function(a,b,c){Gd[a]=Gd[a]||{};var d=1!==c?Nd(b):Hd.get(b);"array"===Pa(d)||"object"===Pa(d)?Gd[a][b]=C(d):Gd[a][b]=d},Td=function(a,b){if(Gd[a])return Gd[a][b]},Ud=function(a,b){Gd[a]&&delete Gd[a][b]};var Vd=function(){var a=!1;return a};var R=function(a,b,c,d){return(2===Wd()||d||"http:"!=D.location.protocol?a:b)+c},Wd=function(){var a=lc(),b;if(1===a)a:{var c=$c;c=c.toLowerCase();for(var d="https://"+c,e="http://"+c,f=1,h=E.getElementsByTagName("script"),k=0;k<h.length&&100>k;k++){var l=h[k].src;if(l){l=l.toLowerCase();if(0===l.indexOf(e)){b=3;break a}1===f&&0===l.indexOf(d)&&(f=2)}}b=f}else b=a;return b};var ke=new RegExp(/^(.*\.)?(google|youtube|blogger|withgoogle)(\.com?)?(\.[a-z]{2})?\.?$/),le={cl:["ecl"],customPixels:["nonGooglePixels"],ecl:["cl"],ehl:["hl"],hl:["ehl"],html:["customScripts","customPixels","nonGooglePixels","nonGoogleScripts","nonGoogleIframes"],customScripts:["html","customPixels","nonGooglePixels","nonGoogleScripts","nonGoogleIframes"],nonGooglePixels:[],nonGoogleScripts:["nonGooglePixels"],nonGoogleIframes:["nonGooglePixels"]},me={cl:["ecl"],customPixels:["customScripts","html"],
ecl:["cl"],ehl:["hl"],hl:["ehl"],html:["customScripts"],customScripts:["html"],nonGooglePixels:["customPixels","customScripts","html","nonGoogleScripts","nonGoogleIframes"],nonGoogleScripts:["customScripts","html"],nonGoogleIframes:["customScripts","html","nonGoogleScripts"]},ne="google customPixels customScripts html nonGooglePixels nonGoogleScripts nonGoogleIframes".split(" ");
var pe=function(a){var b=Ld("gtm.whitelist");b&&P("GTM",9);var c=b&&La(Ea(b),le),d=Ld("gtm.blacklist");d||(d=Ld("tagTypeBlacklist"))&&P("GTM",3);d?
P("GTM",8):d=[];oe()&&(d=Ea(d),d.push("nonGooglePixels","nonGoogleScripts","sandboxedScripts"));0<=q(Ea(d),"google")&&P("GTM",2);var e=d&&La(Ea(d),me),f={};return function(h){var k=h&&h[Ib.ra];if(!k||"string"!=typeof k)return!0;k=k.replace(/^_*/,"");if(void 0!==f[k])return f[k];var l=gd[k]||[],m=a(k,l);if(b){var n;if(n=m)a:{if(0>q(c,k))if(l&&0<l.length)for(var r=0;r<
l.length;r++){if(0>q(c,l[r])){P("GTM",11);n=!1;break a}}else{n=!1;break a}n=!0}m=n}var u=!1;if(d){var p=0<=q(e,k);if(p)u=p;else{var t=za(e,l||[]);t&&P("GTM",10);u=t}}var v=!m||u;v||!(0<=q(l,"sandboxedScripts"))||c&&-1!==q(c,"sandboxedScripts")||(v=za(e,ne));return f[k]=v}},oe=function(){return ke.test(D.location&&D.location.hostname)};var qe={Of:function(a,b){b[Ib.rd]&&"string"===typeof a&&(a=1==b[Ib.rd]?a.toLowerCase():a.toUpperCase());b.hasOwnProperty(Ib.td)&&null===a&&(a=b[Ib.td]);b.hasOwnProperty(Ib.vd)&&void 0===a&&(a=b[Ib.vd]);b.hasOwnProperty(Ib.ud)&&!0===a&&(a=b[Ib.ud]);b.hasOwnProperty(Ib.sd)&&!1===a&&(a=b[Ib.sd]);return a}};var re={active:!0,isWhitelisted:function(){return!0}},se=function(a){var b=I.zones;!b&&a&&(b=I.zones=a());return b};var te=function(){};var ue=!1,ve=0,we=[];function xe(a){if(!ue){var b=E.createEventObject,c="complete"==E.readyState,d="interactive"==E.readyState;if(!a||"readystatechange"!=a.type||c||!b&&d){ue=!0;for(var e=0;e<we.length;e++)G(we[e])}we.push=function(){for(var f=0;f<arguments.length;f++)G(arguments[f]);return 0}}}function ye(){if(!ue&&140>ve){ve++;try{E.documentElement.doScroll("left"),xe()}catch(a){D.setTimeout(ye,50)}}}var ze=function(a){ue?a():we.push(a)};var Ae={},Be={},Ce=function(a,b,c,d){if(!Be[a]||Yc[b]||"__zone"===b)return-1;var e={};Ra(d)&&(e=C(d,e));e.id=c;e.status="timeout";return Be[a].tags.push(e)-1},De=function(a,b,c,d){if(Be[a]){var e=Be[a].tags[b];e&&(e.status=c,e.executionTime=d)}};function Ee(a){for(var b=Ae[a]||[],c=0;c<b.length;c++)b[c]();Ae[a]={push:function(d){d(Wc.s,Be[a])}}}
var He=function(a,b,c){Be[a]={tags:[]};ra(b)&&Fe(a,b);c&&D.setTimeout(function(){return Ee(a)},Number(c));return Ge(a)},Fe=function(a,b){Ae[a]=Ae[a]||[];Ae[a].push(Ia(function(){return G(function(){b(Wc.s,Be[a])})}))};function Ge(a){var b=0,c=0,d=!1;return{add:function(){c++;return Ia(function(){b++;d&&b>=c&&Ee(a)})},Af:function(){d=!0;b>=c&&Ee(a)}}};var Ie=function(){function a(d){return!sa(d)||0>d?0:d}if(!I._li&&D.performance&&D.performance.timing){var b=D.performance.timing.navigationStart,c=sa(Md.get("gtm.start"))?Md.get("gtm.start"):0;I._li={cst:a(c-b),cbt:a(cd-b)}}};var Me={},Ne=function(){return D.GoogleAnalyticsObject&&D[D.GoogleAnalyticsObject]},Oe=!1;
var Pe=function(a){D.GoogleAnalyticsObject||(D.GoogleAnalyticsObject=a||"ga");var b=D.GoogleAnalyticsObject;if(D[b])D.hasOwnProperty(b)||P("GTM",12);else{var c=function(){c.q=c.q||[];c.q.push(arguments)};c.l=Number(new Date);D[b]=c}Ie();return D[b]},Qe=function(a,b,c,d){b=String(b).replace(/\s+/g,"").split(",");var e=Ne();e(a+"require","linker");e(a+"linker:autoLink",b,c,d)};
var Se=function(a){},Re=function(){return D.GoogleAnalyticsObject||"ga"};var Ue=/^(?:(?:https?|mailto|ftp):|[^:/?#]*(?:[/?#]|$))/i;var Ve=/:[0-9]+$/,We=function(a,b,c){for(var d=a.split("&"),e=0;e<d.length;e++){var f=d[e].split("=");if(decodeURIComponent(f[0]).replace(/\+/g," ")===b){var h=f.slice(1).join("=");return c?h:decodeURIComponent(h).replace(/\+/g," ")}}},Ze=function(a,b,c,d,e){b&&(b=String(b).toLowerCase());if("protocol"===b||"port"===b)a.protocol=Xe(a.protocol)||Xe(D.location.protocol);"port"===b?a.port=String(Number(a.hostname?a.port:D.location.port)||("http"==a.protocol?80:"https"==a.protocol?443:"")):"host"===b&&
(a.hostname=(a.hostname||D.location.hostname).replace(Ve,"").toLowerCase());var f=b,h,k=Xe(a.protocol);f&&(f=String(f).toLowerCase());switch(f){case "url_no_fragment":h=Ye(a);break;case "protocol":h=k;break;case "host":h=a.hostname.replace(Ve,"").toLowerCase();if(c){var l=/^www\d*\./.exec(h);l&&l[0]&&(h=h.substr(l[0].length))}break;case "port":h=String(Number(a.port)||("http"==k?80:"https"==k?443:""));break;case "path":a.pathname||a.hostname||P("TAGGING",1);h="/"==a.pathname.substr(0,1)?a.pathname:
"/"+a.pathname;var m=h.split("/");0<=q(d||[],m[m.length-1])&&(m[m.length-1]="");h=m.join("/");break;case "query":h=a.search.replace("?","");e&&(h=We(h,e,void 0));break;case "extension":var n=a.pathname.split(".");h=1<n.length?n[n.length-1]:"";h=h.split("/")[0];break;case "fragment":h=a.hash.replace("#","");break;default:h=a&&a.href}return h},Xe=function(a){return a?a.replace(":","").toLowerCase():""},Ye=function(a){var b="";if(a&&a.href){var c=a.href.indexOf("#");b=0>c?a.href:a.href.substr(0,c)}return b},
$e=function(a){var b=E.createElement("a");a&&(b.href=a);var c=b.pathname;"/"!==c[0]&&(a||P("TAGGING",1),c="/"+c);var d=b.hostname.replace(Ve,"");return{href:b.href,protocol:b.protocol,host:b.host,hostname:d,pathname:c,search:b.search,hash:b.hash,port:b.port}};function ef(a,b,c,d){var e=ub[a],f=ff(a,b,c,d);if(!f)return null;var h=Db(e[Ib.Md],c,[]);if(h&&h.length){var k=h[0];f=ef(k.index,{B:f,w:1===k.he?b.terminate:f,terminate:b.terminate},c,d)}return f}
function ff(a,b,c,d){function e(){if(f[Ib.Xe])k();else{var w=Eb(f,c,[]),y=Ce(c.id,String(f[Ib.ra]),Number(f[Ib.Od]),w[Ib.Ye]),x=!1;w.vtp_gtmOnSuccess=function(){if(!x){x=!0;var A=Ga()-z;Ed(c.id,ub[a],"5");De(c.id,y,"success",A);h()}};w.vtp_gtmOnFailure=function(){if(!x){x=!0;var A=Ga()-z;Ed(c.id,ub[a],"6");De(c.id,y,"failure",A);k()}};w.vtp_gtmTagId=f.tag_id;
w.vtp_gtmEventId=c.id;Ed(c.id,f,"1");var B=function(){var A=Ga()-z;Ed(c.id,f,"7");De(c.id,y,"exception",A);x||(x=!0,k())};var z=Ga();try{Cb(w,c)}catch(A){B(A)}}}var f=ub[a],h=b.B,k=b.w,l=b.terminate;if(c.Lc(f))return null;var m=Db(f[Ib.Pd],c,[]);if(m&&m.length){var n=m[0],r=ef(n.index,{B:h,w:k,terminate:l},c,d);if(!r)return null;h=r;k=2===n.he?l:r}if(f[Ib.Ed]||f[Ib.bf]){var u=f[Ib.Ed]?wb:c.bh,p=h,t=k;if(!u[a]){e=Ia(e);var v=gf(a,u,e);h=v.B;k=v.w}return function(){u[a](p,t)}}return e}
function gf(a,b,c){var d=[],e=[];b[a]=hf(d,e,c);return{B:function(){b[a]=jf;for(var f=0;f<d.length;f++)d[f]()},w:function(){b[a]=kf;for(var f=0;f<e.length;f++)e[f]()}}}function hf(a,b,c){return function(d,e){a.push(d);b.push(e);c()}}function jf(a){a()}function kf(a,b){b()};var nf=function(a,b){for(var c=[],d=0;d<ub.length;d++)if(a.kb[d]){var e=ub[d];var f=b.add();try{var h=ef(d,{B:f,w:f,terminate:f},a,d);h?c.push({Fe:d,Ae:Fb(e),Zf:h}):(lf(d,a),f())}catch(l){f()}}b.Af();c.sort(mf);for(var k=0;k<c.length;k++)c[k].Zf();return 0<c.length};function mf(a,b){var c,d=b.Ae,e=a.Ae;c=d>e?1:d<e?-1:0;var f;if(0!==c)f=c;else{var h=a.Fe,k=b.Fe;f=h>k?1:h<k?-1:0}return f}
function lf(a,b){if(!Bd)return;var c=function(d){var e=b.Lc(ub[d])?"3":"4",f=Db(ub[d][Ib.Md],b,[]);f&&f.length&&c(f[0].index);Ed(b.id,ub[d],e);var h=Db(ub[d][Ib.Pd],b,[]);h&&h.length&&c(h[0].index)};c(a);}
var of=!1,pf=function(a,b,c,d,e){if("gtm.js"==b){if(of)return!1;of=!0}Dd(a,b);var f=He(a,d,e);Sd(a,"event",1);Sd(a,"ecommerce",1);Sd(a,"gtm");var h={id:a,name:b,Lc:pe(c),kb:[],bh:[],ue:function(){P("GTM",6)}};h.kb=Mb(h);var k=nf(h,f);"gtm.js"!==b&&"gtm.sync"!==b||Se(Wc.s);if(!k)return k;for(var l=0;l<h.kb.length;l++)if(h.kb[l]){var m=ub[l];if(m&&!Yc[String(m[Ib.ra])])return!0}return!1};var rf=/^https?:\/\/www\.googletagmanager\.com/;function sf(){var a;return a}function uf(a,b){}
function tf(a){0!==a.indexOf("http://")&&0!==a.indexOf("https://")&&(a="https://"+a);"/"===a[a.length-1]&&(a=a.substring(0,a.length-1));return a}function vf(){var a=!1;return a};var wf=function(){this.eventModel={};this.targetConfig={};this.containerConfig={};this.h={};this.globalConfig={};this.B=function(){};this.w=function(){}},xf=function(a){var b=new wf;b.eventModel=a;return b},yf=function(a,b){a.targetConfig=b;return a},zf=function(a,b){a.containerConfig=b;return a},Af=function(a,b){a.h=b;return a},Bf=function(a,b){a.globalConfig=b;return a},Cf=function(a,b){a.B=b;return a},Df=function(a,b){a.w=b;return a};
wf.prototype.getWithConfig=function(a){if(void 0!==this.eventModel[a])return this.eventModel[a];if(void 0!==this.targetConfig[a])return this.targetConfig[a];if(void 0!==this.containerConfig[a])return this.containerConfig[a];if(void 0!==this.h[a])return this.h[a];if(void 0!==this.globalConfig[a])return this.globalConfig[a]};
var Ef=function(a){function b(e){Aa(e,function(f){c[f]=null})}var c={};b(a.eventModel);b(a.targetConfig);b(a.containerConfig);b(a.globalConfig);var d=[];Aa(c,function(e){d.push(e)});return d};var Ff=function(a,b,c){for(var d=[],e=String(b||document.cookie).split(";"),f=0;f<e.length;f++){var h=e[f].split("="),k=h[0].replace(/^\s*|\s*$/g,"");if(k&&k==a){var l=h.slice(1).join("=").replace(/^\s*|\s*$/g,"");l&&c&&(l=decodeURIComponent(l));d.push(l)}}return d},If=function(a,b,c,d){var e=Gf(a,d);if(1===e.length)return e[0].id;if(0!==e.length){e=Hf(e,function(f){return f.Jb},b);if(1===e.length)return e[0].id;e=Hf(e,function(f){return f.lb},c);return e[0]?e[0].id:void 0}};
function Jf(a,b,c){var d=document.cookie;document.cookie=a;var e=document.cookie;return d!=e||void 0!=c&&0<=Ff(b,e).indexOf(c)}
var Mf=function(a,b,c,d,e,f){d=d||"auto";var h={path:c||"/"};e&&(h.expires=e);"none"!==d&&(h.domain=d);var k;a:{var l=b,m;if(void 0==l)m=a+"=deleted; expires="+(new Date(0)).toUTCString();else{f&&(l=encodeURIComponent(l));var n=l;n&&1200<n.length&&(n=n.substring(0,1200));l=n;m=a+"="+l}var r=void 0,u=void 0,p;for(p in h)if(h.hasOwnProperty(p)){var t=h[p];if(null!=t)switch(p){case "secure":t&&(m+="; secure");break;case "domain":r=t;break;default:"path"==p&&(u=t),"expires"==p&&t instanceof Date&&(t=
t.toUTCString()),m+="; "+p+"="+t}}if("auto"===r){for(var v=Kf(),w=0;w<v.length;++w){var y="none"!=v[w]?v[w]:void 0;if(!Lf(y,u)&&Jf(m+(y?"; domain="+y:""),a,l)){k=!0;break a}}k=!1}else r&&"none"!=r&&(m+="; domain="+r),k=!Lf(r,u)&&Jf(m,a,l)}return k};function Hf(a,b,c){for(var d=[],e=[],f,h=0;h<a.length;h++){var k=a[h],l=b(k);l===c?d.push(k):void 0===f||l<f?(e=[k],f=l):l===f&&e.push(k)}return 0<d.length?d:e}
function Gf(a,b){for(var c=[],d=Ff(a),e=0;e<d.length;e++){var f=d[e].split("."),h=f.shift();if(!b||-1!==b.indexOf(h)){var k=f.shift();k&&(k=k.split("-"),c.push({id:f.join("."),Jb:1*k[0]||1,lb:1*k[1]||1}))}}return c}
var Nf=/^(www\.)?google(\.com?)?(\.[a-z]{2})?$/,Of=/(^|\.)doubleclick\.net$/i,Lf=function(a,b){return Of.test(document.location.hostname)||"/"===b&&Nf.test(a)},Kf=function(){var a=[],b=document.location.hostname.split(".");if(4===b.length){var c=b[b.length-1];if(parseInt(c,10).toString()===c)return["none"]}for(var d=b.length-2;0<=d;d--)a.push(b.slice(d).join("."));var e=document.location.hostname;Of.test(e)||Nf.test(e)||a.push("none");return a};function Pf(){for(var a=Qf,b={},c=0;c<a.length;++c)b[a[c]]=c;return b}function Rf(){var a="ABCDEFGHIJKLMNOPQRSTUVWXYZ";a+=a.toLowerCase()+"0123456789-_";return a+"."}var Qf,Sf;function Tf(a){Qf=Qf||Rf();Sf=Sf||Pf();for(var b=[],c=0;c<a.length;c+=3){var d=c+1<a.length,e=c+2<a.length,f=a.charCodeAt(c),h=d?a.charCodeAt(c+1):0,k=e?a.charCodeAt(c+2):0,l=f>>2,m=(f&3)<<4|h>>4,n=(h&15)<<2|k>>6,r=k&63;e||(r=64,d||(n=64));b.push(Qf[l],Qf[m],Qf[n],Qf[r])}return b.join("")}
function Uf(a){function b(l){for(;d<a.length;){var m=a.charAt(d++),n=Sf[m];if(null!=n)return n;if(!/^[\s\xa0]*$/.test(m))throw Error("Unknown base64 encoding at char: "+m);}return l}Qf=Qf||Rf();Sf=Sf||Pf();for(var c="",d=0;;){var e=b(-1),f=b(0),h=b(64),k=b(64);if(64===k&&-1===e)return c;c+=String.fromCharCode(e<<2|f>>4);64!=h&&(c+=String.fromCharCode(f<<4&240|h>>2),64!=k&&(c+=String.fromCharCode(h<<6&192|k)))}};var Vf;var Zf=function(){var a=Wf,b=Xf,c=Yf(),d=function(h){a(h.target||h.srcElement||{})},e=function(h){b(h.target||h.srcElement||{})};if(!c.init){oc(E,"mousedown",d);oc(E,"keyup",d);oc(E,"submit",e);var f=HTMLFormElement.prototype.submit;HTMLFormElement.prototype.submit=function(){b(this);f.call(this)};c.init=!0}},$f=function(a,b,c){for(var d=Yf().decorators,e={},f=0;f<d.length;++f){var h=d[f],k;if(k=!c||h.forms)a:{var l=h.domains,m=a;if(l&&(h.sameHost||m!==E.location.hostname))for(var n=0;n<l.length;n++)if(l[n]instanceof
RegExp){if(l[n].test(m)){k=!0;break a}}else if(0<=m.indexOf(l[n])){k=!0;break a}k=!1}if(k){var r=h.placement;void 0==r&&(r=h.fragment?2:1);r===b&&Ja(e,h.callback())}}return e},Yf=function(){var a=ic("google_tag_data",{}),b=a.gl;b&&b.decorators||(b={decorators:[]},a.gl=b);return b};var ag=/(.*?)\*(.*?)\*(.*)/,cg=/^https?:\/\/([^\/]*?)\.?cdn\.ampproject\.org\/?(.*)/,dg=/^(?:www\.|m\.|amp\.)+/,eg=/([^?#]+)(\?[^#]*)?(#.*)?/;function fg(a){return new RegExp("(.*?)(^|&)"+a+"=([^&]*)&?(.*)")}
var hg=function(a){var b=[],c;for(c in a)if(a.hasOwnProperty(c)){var d=a[c];void 0!==d&&d===d&&null!==d&&"[object Object]"!==d.toString()&&(b.push(c),b.push(Tf(String(d))))}var e=b.join("*");return["1",gg(e),e].join("*")},gg=function(a,b){var c=[window.navigator.userAgent,(new Date).getTimezoneOffset(),window.navigator.userLanguage||window.navigator.language,Math.floor((new Date).getTime()/60/1E3)-(void 0===b?0:b),a].join("*"),d;if(!(d=Vf)){for(var e=Array(256),f=0;256>f;f++){for(var h=f,k=0;8>k;k++)h=
h&1?h>>>1^3988292384:h>>>1;e[f]=h}d=e}Vf=d;for(var l=4294967295,m=0;m<c.length;m++)l=l>>>8^Vf[(l^c.charCodeAt(m))&255];return((l^-1)>>>0).toString(36)},jg=function(){return function(a){var b=$e(D.location.href),c=b.search.replace("?",""),d=We(c,"_gl",!0)||"";a.query=ig(d)||{};var e=Ze(b,"fragment").match(fg("_gl"));a.fragment=ig(e&&e[3]||"")||{}}},kg=function(){var a=jg(),b=Yf();b.data||(b.data={query:{},fragment:{}},a(b.data));var c={},d=b.data;d&&(Ja(c,d.query),Ja(c,d.fragment));return c},ig=function(a){var b;
b=void 0===b?3:b;try{if(a){var c;a:{for(var d=a,e=0;3>e;++e){var f=ag.exec(d);if(f){c=f;break a}d=decodeURIComponent(d)}c=void 0}var h=c;if(h&&"1"===h[1]){var k=h[3],l;a:{for(var m=h[2],n=0;n<b;++n)if(m===gg(k,n)){l=!0;break a}l=!1}if(l){for(var r={},u=k?k.split("*"):[],p=0;p<u.length;p+=2)r[u[p]]=Uf(u[p+1]);return r}}}}catch(t){}};
function lg(a,b,c,d){function e(n){var r=n,u=fg(a).exec(r),p=r;if(u){var t=u[2],v=u[4];p=u[1];v&&(p=p+t+v)}n=p;var w=n.charAt(n.length-1);n&&"&"!==w&&(n+="&");return n+m}d=void 0===d?!1:d;var f=eg.exec(c);if(!f)return"";var h=f[1],k=f[2]||"",l=f[3]||"",m=a+"="+b;d?l="#"+e(l.substring(1)):k="?"+e(k.substring(1));return""+h+k+l}
function mg(a,b){var c="FORM"===(a.tagName||"").toUpperCase(),d=$f(b,1,c),e=$f(b,2,c),f=$f(b,3,c);if(Ka(d)){var h=hg(d);c?ng("_gl",h,a):og("_gl",h,a,!1)}if(!c&&Ka(e)){var k=hg(e);og("_gl",k,a,!0)}for(var l in f)if(f.hasOwnProperty(l))a:{var m=l,n=f[l],r=a;if(r.tagName){if("a"===r.tagName.toLowerCase()){og(m,n,r,void 0);break a}if("form"===r.tagName.toLowerCase()){ng(m,n,r);break a}}"string"==typeof r&&lg(m,n,r,void 0)}}
function og(a,b,c,d){if(c.href){var e=lg(a,b,c.href,void 0===d?!1:d);Ue.test(e)&&(c.href=e)}}
function ng(a,b,c){if(c&&c.action){var d=(c.method||"").toLowerCase();if("get"===d){for(var e=c.childNodes||[],f=!1,h=0;h<e.length;h++){var k=e[h];if(k.name===a){k.setAttribute("value",b);f=!0;break}}if(!f){var l=E.createElement("input");l.setAttribute("type","hidden");l.setAttribute("name",a);l.setAttribute("value",b);c.appendChild(l)}}else if("post"===d){var m=lg(a,b,c.action);Ue.test(m)&&(c.action=m)}}}
var Wf=function(a){try{var b;a:{for(var c=a,d=100;c&&0<d;){if(c.href&&c.nodeName.match(/^a(?:rea)?$/i)){b=c;break a}c=c.parentNode;d--}b=null}var e=b;if(e){var f=e.protocol;"http:"!==f&&"https:"!==f||mg(e,e.hostname)}}catch(h){}},Xf=function(a){try{if(a.action){var b=Ze($e(a.action),"host");mg(a,b)}}catch(c){}},pg=function(a,b,c,d){Zf();var e="fragment"===c?2:1,f={callback:a,domains:b,fragment:2===e,placement:e,forms:!!d,sameHost:!1};Yf().decorators.push(f)},qg=function(){var a=E.location.hostname,
b=cg.exec(E.referrer);if(!b)return!1;var c=b[2],d=b[1],e="";if(c){var f=c.split("/"),h=f[1];e="s"===h?decodeURIComponent(f[2]):decodeURIComponent(h)}else if(d){if(0===d.indexOf("xn--"))return!1;e=d.replace(/-/g,".").replace(/\.\./g,"-")}var k=a.replace(dg,""),l=e.replace(dg,""),m;if(!(m=k===l)){var n="."+l;m=k.substring(k.length-n.length,k.length)===n}return m},rg=function(a,b){return!1===a?!1:a||b||qg()};var sg={};var tg=/^\w+$/,ug=/^[\w-]+$/,vg=/^~?[\w-]+$/,wg={aw:"_aw",dc:"_dc",gf:"_gf",ha:"_ha",gp:"_gp"};function xg(a){return a&&"string"==typeof a&&a.match(tg)?a:"_gcl"}
var zg=function(){var a=$e(D.location.href),b=Ze(a,"query",!1,void 0,"gclid"),c=Ze(a,"query",!1,void 0,"gclsrc"),d=Ze(a,"query",!1,void 0,"dclid");if(!b||!c){var e=a.hash.replace("#","");b=b||We(e,"gclid",void 0);c=c||We(e,"gclsrc",void 0)}return yg(b,c,d)},yg=function(a,b,c){var d={},e=function(f,h){d[h]||(d[h]=[]);d[h].push(f)};d.gclid=a;d.gclsrc=b;d.dclid=c;if(void 0!==a&&a.match(ug))switch(b){case void 0:e(a,"aw");break;case "aw.ds":e(a,"aw");e(a,"dc");break;case "ds":e(a,"dc");break;case "3p.ds":(void 0==
sg.gtm_3pds?0:sg.gtm_3pds)&&e(a,"dc");break;case "gf":e(a,"gf");break;case "ha":e(a,"ha");break;case "gp":e(a,"gp")}c&&e(c,"dc");return d},Bg=function(a){var b=zg();Ag(b,a)};
function Ag(a,b,c){function d(r,u){var p=Cg(r,e);p&&Mf(p,u,h,f,l,!0)}b=b||{};var e=xg(b.prefix),f=b.domain||"auto",h=b.path||"/",k=void 0==b.Ka?7776E3:b.Ka;c=c||Ga();var l=0==k?void 0:new Date(c+1E3*k),m=Math.round(c/1E3),n=function(r){return["GCL",m,r].join(".")};a.aw&&(!0===b.Mh?d("aw",n("~"+a.aw[0])):d("aw",n(a.aw[0])));a.dc&&d("dc",n(a.dc[0]));a.gf&&d("gf",n(a.gf[0]));a.ha&&d("ha",n(a.ha[0]));a.gp&&d("gp",n(a.gp[0]))}
var Eg=function(a,b,c,d,e){for(var f=kg(),h=xg(b),k=0;k<a.length;++k){var l=a[k];if(void 0!==wg[l]){var m=Cg(l,h),n=f[m];if(n){var r=Math.min(Dg(n),Ga()),u;b:{for(var p=r,t=Ff(m,E.cookie),v=0;v<t.length;++v)if(Dg(t[v])>p){u=!0;break b}u=!1}u||Mf(m,n,c,d,0==e?void 0:new Date(r+1E3*(null==e?7776E3:e)),!0)}}}var w={prefix:b,path:c,domain:d};Ag(yg(f.gclid,f.gclsrc),w)},Cg=function(a,b){var c=wg[a];if(void 0!==c)return b+c},Dg=function(a){var b=a.split(".");return 3!==b.length||"GCL"!==b[0]?0:1E3*(Number(b[1])||
0)};function Fg(a){var b=a.split(".");if(3==b.length&&"GCL"==b[0]&&b[1])return b[2]}
var Gg=function(a,b,c,d,e){if(va(b)){var f=xg(e);pg(function(){for(var h={},k=0;k<a.length;++k){var l=Cg(a[k],f);if(l){var m=Ff(l,E.cookie);m.length&&(h[l]=m.sort()[m.length-1])}}return h},b,c,d)}},Hg=function(a){return a.filter(function(b){return vg.test(b)})},Ig=function(a,b){for(var c=xg(b&&b.prefix),d={},e=0;e<a.length;e++)wg[a[e]]&&(d[a[e]]=wg[a[e]]);Aa(d,function(f,h){var k=Ff(c+h,E.cookie);if(k.length){var l=k[0],m=Dg(l),n={};n[f]=[Fg(l)];Ag(n,b,m)}})};function Jg(){var a=zg(),b=a.gclid,c=a.gclsrc;if(b&&(!c||"aw.ds"===c)){var d;I.reported_gclid||(I.reported_gclid={});d=I.reported_gclid;if(!d[b]){d[b]=!0;var e="/pagead/landing?gclid="+encodeURIComponent(b);c&&(e+="&gclsrc="+encodeURIComponent(c));vc("https://www.google.com"+e)}}};var Kg;if(3===Wc.yb.length)Kg="g";else{var Lg="G";Kg=Lg}
var Mg={"":"n",UA:"u",AW:"a",DC:"d",G:"e",GF:"f",HA:"h",GTM:Kg,OPT:"o"},Ng=function(a){var b=Wc.s.split("-"),c=b[0].toUpperCase(),d=Mg[c]||"i",e=a&&"GTM"===c?b[1]:"OPT"===c?b[1]:"",f;if(3===Wc.yb.length){var h=void 0;f="2"+(h||"w")}else f=
"";return f+d+Wc.yb+e};var Xg=function(){for(var a=gc.userAgent+(E.cookie||"")+(E.referrer||""),b=a.length,c=D.history.length;0<c;)a+=c--^b++;var d=1,e,f,h;if(a)for(d=0,f=a.length-1;0<=f;f--)h=a.charCodeAt(f),d=(d<<6&268435455)+h+(h<<14),e=d&266338304,d=0!=e?d^e>>21:d;return[Math.round(2147483647*Math.random())^d&2147483647,Math.round(Ga()/1E3)].join(".")},$g=function(a,b,c,d){var e=Yg(b);return If(a,e,Zg(c),d)},ah=function(a,b,c,d){var e=""+Yg(c),f=Zg(d);1<f&&(e+="-"+f);return[b,e,a].join(".")},Yg=function(a){if(!a)return 1;
a=0===a.indexOf(".")?a.substr(1):a;return a.split(".").length},Zg=function(a){if(!a||"/"===a)return 1;"/"!==a[0]&&(a="/"+a);"/"!==a[a.length-1]&&(a+="/");return a.split("/").length-1};var bh=["1"],ch={},gh=function(a,b,c,d){var e=dh(a);ch[e]||eh(e,b,c)||(fh(e,Xg(),b,c,d),eh(e,b,c))};function fh(a,b,c,d,e){var f=ah(b,"1",d,c);Mf(a,f,c,d,0==e?void 0:new Date(Ga()+1E3*(void 0==e?7776E3:e)))}function eh(a,b,c){var d=$g(a,b,c,bh);d&&(ch[a]=d);return d}function dh(a){return(a||"_gcl")+"_au"};var hh=function(){for(var a=[],b=E.cookie.split(";"),c=/^\s*_gac_(UA-\d+-\d+)=\s*(.+?)\s*$/,d=0;d<b.length;d++){var e=b[d].match(c);e&&a.push({ed:e[1],value:e[2]})}var f={};if(!a||!a.length)return f;for(var h=0;h<a.length;h++){var k=a[h].value.split(".");"1"==k[0]&&3==k.length&&k[1]&&(f[a[h].ed]||(f[a[h].ed]=[]),f[a[h].ed].push({timestamp:k[1],bg:k[2]}))}return f};var ih=/^\d+\.fls\.doubleclick\.net$/;function jh(a){var b=$e(D.location.href),c=Ze(b,"host",!1);if(c&&c.match(ih)){var d=Ze(b,"path").split(a+"=");if(1<d.length)return d[1].split(";")[0].split("?")[0]}}
function kh(a,b){if("aw"==a||"dc"==a){var c=jh("gcl"+a);if(c)return c.split(".")}var d=xg(b);if("_gcl"==d){var e;e=zg()[a]||[];if(0<e.length)return e}var f=Cg(a,d),h;if(f){var k=[];if(E.cookie){var l=Ff(f,E.cookie);if(l&&0!=l.length){for(var m=0;m<l.length;m++){var n=Fg(l[m]);n&&-1===q(k,n)&&k.push(n)}h=Hg(k)}else h=k}else h=k}else h=[];return h}
var lh=function(){var a=jh("gac");if(a)return decodeURIComponent(a);var b=hh(),c=[];Aa(b,function(d,e){for(var f=[],h=0;h<e.length;h++)f.push(e[h].bg);f=Hg(f);f.length&&c.push(d+":"+f.join(","))});return c.join(";")},mh=function(a,b,c,d,e){gh(b,c,d,e);var f=ch[dh(b)],h=zg().dc||[],k=!1;if(f&&0<h.length){var l=I.joined_au=I.joined_au||{},m=b||"_gcl";if(!l[m])for(var n=0;n<h.length;n++){var r="https://adservice.google.com/ddm/regclk";r=r+"?gclid="+h[n]+"&auiddc="+f;vc(r);k=l[m]=!0}}null==a&&(a=k);if(a&&f){var u=dh(b),
p=ch[u];p&&fh(u,p,c,d,e)}};var ei={},fi=["G","GP"];ei.Ge="";var gi=ei.Ge.split(",");function hi(){var a=I;return a.gcq=a.gcq||new ii}
var ji=function(a,b,c){hi().register(a,b,c)},ki=function(a,b,c,d){hi().push("event",[b,a],c,d)},li=function(a,b){hi().push("config",[a],b)},mi={},ni=function(){this.status=1;this.containerConfig={};this.targetConfig={};this.i={};this.m=null;this.h=!1},oi=function(a,b,c,d,e){this.type=a;this.m=b;this.N=c||"";this.h=d;this.i=e},ii=function(){this.i={};this.m={};this.h=[]},pi=function(a,b){var c=Tc(b);return a.i[c.containerId]=a.i[c.containerId]||new ni},qi=function(a,b,c,d){if(d.N){var e=pi(a,d.N),
f=e.m;if(f){var h=C(c),k=C(e.targetConfig[d.N]),l=C(e.containerConfig),m=C(e.i),n=C(a.m),r=Ld("gtm.uniqueEventId"),u=Tc(d.N).prefix,p=Df(Cf(Bf(Af(zf(yf(xf(h),k),l),m),n),function(){Fd(r,u,"2");}),function(){Fd(r,u,"3");});try{Fd(r,u,"1");f(d.N,b,d.m,p)}catch(t){
Fd(r,u,"4");}}}};
ii.prototype.register=function(a,b,c){if(3!==pi(this,a).status){pi(this,a).m=b;pi(this,a).status=3;c&&(pi(this,a).i=c);var d=Tc(a),e=mi[d.containerId];if(void 0!==e){var f=I[d.containerId].bootstrap,h=d.prefix.toUpperCase();I[d.containerId]._spx&&(h=h.toLowerCase());var k=Ld("gtm.uniqueEventId"),l=h,m=Ga()-f;if(Bd&&!sd[k]){k!==od&&(md(),od=k);var n=l+"."+Math.floor(f-e)+"."+Math.floor(m);xd=xd?xd+","+n:"&cl="+n}delete mi[d.containerId]}this.flush()}};
ii.prototype.push=function(a,b,c,d){var e=Math.floor(Ga()/1E3);a:if(c){var f=Tc(c),h;if(h=f){var k;if(k=1===pi(this,c).status)b:{var l=f.prefix;k=!0}h=k}if(h)if(pi(this,c).status=2,this.push("require",[],f.containerId),mi[f.containerId]=Ga(),Vd()){}else{var n=encodeURIComponent(f.containerId),r=("http:"!=D.location.protocol?"https:":"http:")+"//www.googletagmanager.com";
kc(r+"/gtag/js?id="+n+"&l=dataLayer&cx=c")}}this.h.push(new oi(a,e,c,b,d));d||this.flush()};
ii.prototype.flush=function(a){for(var b=this;this.h.length;){var c=this.h[0];if(c.i)c.i=!1,this.h.push(c);else switch(c.type){case "require":if(3!==pi(this,c.N).status&&!a)return;break;case "set":Aa(c.h[0],function(l,m){C(Ma(l,m),b.m)});break;case "config":var d=c.h[0],e=!!d[H.Qb];delete d[H.Qb];var f=pi(this,c.N),h=Tc(c.N),k=h.containerId===h.id;e||(k?f.containerConfig={}:f.targetConfig[c.N]={});f.h&&e||qi(this,H.D,d,c);f.h=!0;delete d[H.qa];k?C(d,f.containerConfig):C(d,f.targetConfig[c.N]);break;
case "event":qi(this,c.h[1],c.h[0],c)}this.h.shift()}};var ri=["GP","G"],si="G".split(/,/);var ti=!1;ti=!0;var ui=null,vi={},wi={},xi;function yi(a,b){var c={event:a};b&&(c.eventModel=C(b),b[H.nc]&&(c.eventCallback=b[H.nc]),b[H.tb]&&(c.eventTimeout=b[H.tb]));return c}
var Ei={config:function(a){},event:function(a){var b=a[1];if(g(b)&&!(3<a.length)){var c;if(2<a.length){if(!Ra(a[2])&&void 0!=a[2])return;c=a[2]}var d=yi(b,c);return d}},js:function(a){if(2==a.length&&a[1].getTime)return{event:"gtm.js","gtm.start":a[1].getTime()}},policy:function(){},set:function(a){var b;2==a.length&&Ra(a[1])?b=C(a[1]):3==a.length&&
g(a[1])&&(b={},Ra(a[2])||va(a[2])?b[a[1]]=C(a[2]):b[a[1]]=a[2]);if(b){b._clear=!0;return b}}},Fi={policy:!0};var Gi=function(a,b){var c=a.hide;if(c&&void 0!==c[b]&&c.end){c[b]=!1;var d=!0,e;for(e in c)if(c.hasOwnProperty(e)&&!0===c[e]){d=!1;break}d&&(c.end(),c.end=null)}},Ii=function(a){var b=Hi(),c=b&&b.hide;c&&c.end&&(c[a]=!0)};var Ji=!1,Ki=[];function Li(){if(!Ji){Ji=!0;for(var a=0;a<Ki.length;a++)G(Ki[a])}}var Mi=function(a){Ji?G(a):Ki.push(a)};var bj=function(a){if(aj(a))return a;this.h=a};bj.prototype.fg=function(){return this.h};var aj=function(a){return!a||"object"!==Pa(a)||Ra(a)?!1:"getUntrustedUpdateValue"in a};bj.prototype.getUntrustedUpdateValue=bj.prototype.fg;var cj=[],dj=!1,ej=function(a){return D["dataLayer"].push(a)},fj=function(a){var b=I["dataLayer"],c=b?b.subscribers:1,d=0;return function(){++d===c&&a()}};
function gj(a){var b=a._clear;Aa(a,function(f,h){"_clear"!==f&&(b&&Rd(f,void 0),Rd(f,h))});bd||(bd=a["gtm.start"]);var c=a.event;if(!c)return!1;var d=a["gtm.uniqueEventId"];d||(d=hd(),a["gtm.uniqueEventId"]=d,Rd("gtm.uniqueEventId",d));dd=c;var e=
hj(a);dd=null;switch(c){case "gtm.init":P("GTM",19),e&&P("GTM",20)}return e}function hj(a){var b=a.event,c=a["gtm.uniqueEventId"],d,e=I.zones;d=e?e.checkState(Wc.s,c):re;return d.active?pf(c,b,d.isWhitelisted,a.eventCallback,a.eventTimeout)?!0:!1:!1}
function ij(){for(var a=!1;!dj&&0<cj.length;){dj=!0;delete Id.eventModel;Kd();var b=cj.shift();if(null!=b){var c=aj(b);if(c){var d=b;b=aj(d)?d.getUntrustedUpdateValue():void 0;for(var e=["gtm.whitelist","gtm.blacklist","tagTypeBlacklist"],f=0;f<e.length;f++){var h=e[f],k=Ld(h,1);if(va(k)||Ra(k))k=C(k);Jd[h]=k}}try{if(ra(b))try{b.call(Md)}catch(v){}else if(va(b)){var l=b;if(g(l[0])){var m=
l[0].split("."),n=m.pop(),r=l.slice(1),u=Ld(m.join("."),2);if(void 0!==u&&null!==u)try{u[n].apply(u,r)}catch(v){}}}else{var p=b;if(p&&("[object Arguments]"==Object.prototype.toString.call(p)||Object.prototype.hasOwnProperty.call(p,"callee"))){a:{if(b.length&&g(b[0])){var t=Ei[b[0]];if(t&&(!c||!Fi[b[0]])){b=t(b);break a}}b=void 0}if(!b){dj=!1;continue}}a=gj(b)||a}}finally{c&&Kd(!0)}}dj=!1}
return!a}function jj(){var a=ij();try{Gi(D["dataLayer"],Wc.s)}catch(b){}return a}
var lj=function(){var a=ic("dataLayer",[]),b=ic("google_tag_manager",{});b=b["dataLayer"]=b["dataLayer"]||{};ze(function(){b.gtmDom||(b.gtmDom=!0,a.push({event:"gtm.dom"}))});Mi(function(){b.gtmLoad||(b.gtmLoad=!0,a.push({event:"gtm.load"}))});b.subscribers=(b.subscribers||0)+1;var c=a.push;a.push=function(){var d;if(0<I.SANDBOXED_JS_SEMAPHORE){d=[];for(var e=0;e<arguments.length;e++)d[e]=new bj(arguments[e])}else d=[].slice.call(arguments,0);var f=c.apply(a,d);cj.push.apply(cj,d);if(300<
this.length)for(P("GTM",4);300<this.length;)this.shift();var h="boolean"!==typeof f||f;return ij()&&h};cj.push.apply(cj,a.slice(0));kj()&&G(jj)},kj=function(){var a=!0;return a};var mj={};mj.ub=new String("undefined");
var nj=function(a){this.h=function(b){for(var c=[],d=0;d<a.length;d++)c.push(a[d]===mj.ub?b:a[d]);return c.join("")}};nj.prototype.toString=function(){return this.h("undefined")};nj.prototype.valueOf=nj.prototype.toString;mj.lf=nj;mj.xc={};mj.Rf=function(a){return new nj(a)};var oj={};mj.Ug=function(a,b){var c=hd();oj[c]=[a,b];return c};mj.ce=function(a){var b=a?0:1;return function(c){var d=oj[c];if(d&&"function"===typeof d[b])d[b]();oj[c]=void 0}};mj.og=function(a){for(var b=!1,c=!1,d=2;d<a.length;d++)b=
b||8===a[d],c=c||16===a[d];return b&&c};mj.Kg=function(a){if(a===mj.ub)return a;var b=hd();mj.xc[b]=a;return'google_tag_manager["'+Wc.s+'"].macro('+b+")"};mj.zg=function(a,b,c){a instanceof mj.lf&&(a=a.h(mj.Ug(b,c)),b=qa);return{Jc:a,B:b}};var pj=function(a,b,c){function d(f,h){var k=f[h];return k}var e={event:b,"gtm.element":a,"gtm.elementClasses":d(a,"className"),"gtm.elementId":a["for"]||qc(a,"id")||"","gtm.elementTarget":a.formTarget||d(a,"target")||""};c&&(e["gtm.triggers"]=c.join(","));e["gtm.elementUrl"]=(a.attributes&&a.attributes.formaction?a.formAction:"")||a.action||d(a,"href")||a.src||a.code||a.codebase||
"";return e},qj=function(a){I.hasOwnProperty("autoEventsSettings")||(I.autoEventsSettings={});var b=I.autoEventsSettings;b.hasOwnProperty(a)||(b[a]={});return b[a]},rj=function(a,b,c){qj(a)[b]=c},sj=function(a,b,c,d){var e=qj(a),f=Ha(e,b,d);e[b]=c(f)},tj=function(a,b,c){var d=qj(a);return Ha(d,b,c)};var uj=["input","select","textarea"],vj=["button","hidden","image","reset","submit"],wj=function(a){var b=a.tagName.toLowerCase();return!wa(uj,function(c){return c===b})||"input"===b&&wa(vj,function(c){return c===a.type.toLowerCase()})?!1:!0},xj=function(a){return a.form?a.form.tagName?a.form:E.getElementById(a.form):uc(a,["form"],100)},yj=function(a,b,c){if(!a.elements)return 0;for(var d=b.getAttribute(c),e=0,f=1;e<a.elements.length;e++){var h=a.elements[e];if(wj(h)){if(h.getAttribute(c)===d)return f;
f++}}return 0};var zj=!!D.MutationObserver,Aj=void 0,Bj=function(a){if(!Aj){var b=function(){var c=E.body;if(c)if(zj)(new MutationObserver(function(){for(var e=0;e<Aj.length;e++)G(Aj[e])})).observe(c,{childList:!0,subtree:!0});else{var d=!1;oc(c,"DOMNodeInserted",function(){d||(d=!0,G(function(){d=!1;for(var e=0;e<Aj.length;e++)G(Aj[e])}))})}};Aj=[];E.body?b():G(b)}Aj.push(a)};var Wj=D.clearTimeout,Xj=D.setTimeout,V=function(a,b,c){if(Vd()){b&&G(b)}else return kc(a,b,c)},Yj=function(){return D.location.href},Zj=function(a){return Ze($e(a),"fragment")},ak=function(a){return Ye($e(a))},W=function(a,b){return Ld(a,b||2)},bk=function(a,b,c){var d;b?(a.eventCallback=b,c&&(a.eventTimeout=c),d=ej(a)):d=ej(a);return d},ck=function(a,b){D[a]=b},X=function(a,b,c){b&&(void 0===D[a]||c&&!D[a])&&(D[a]=
b);return D[a]},dk=function(a,b,c){return Ff(a,b,void 0===c?!0:!!c)},ek=function(a,b){if(Vd()){b&&G(b)}else mc(a,b)},fk=function(a){return!!tj(a,"init",!1)},gk=function(a){rj(a,"init",!0)},hk=function(a,b){var c=(void 0===b?0:b)?"www.googletagmanager.com/gtag/js":$c;c+="?id="+encodeURIComponent(a)+"&l=dataLayer";V(R("https://","http://",c))},ik=function(a,b){var c=a[b];return c};
var jk=mj.zg;var Gk=new ya;function Hk(a,b){function c(h){var k=$e(h),l=Ze(k,"protocol"),m=Ze(k,"host",!0),n=Ze(k,"port"),r=Ze(k,"path").toLowerCase().replace(/\/$/,"");if(void 0===l||"http"==l&&"80"==n||"https"==l&&"443"==n)l="web",n="default";return[l,m,n,r]}for(var d=c(String(a)),e=c(String(b)),f=0;f<d.length;f++)if(d[f]!==e[f])return!1;return!0}
function Ik(a){return Jk(a)?1:0}
function Jk(a){var b=a.arg0,c=a.arg1;if(a.any_of&&va(c)){for(var d=0;d<c.length;d++)if(Ik({"function":a["function"],arg0:b,arg1:c[d]}))return!0;return!1}switch(a["function"]){case "_cn":return 0<=String(b).indexOf(String(c));case "_css":var e;a:{if(b){var f=["matches","webkitMatchesSelector","mozMatchesSelector","msMatchesSelector","oMatchesSelector"];try{for(var h=0;h<f.length;h++)if(b[f[h]]){e=b[f[h]](c);break a}}catch(v){}}e=!1}return e;case "_ew":var k,l;k=String(b);l=String(c);var m=k.length-
l.length;return 0<=m&&k.indexOf(l,m)==m;case "_eq":return String(b)==String(c);case "_ge":return Number(b)>=Number(c);case "_gt":return Number(b)>Number(c);case "_lc":var n;n=String(b).split(",");return 0<=q(n,String(c));case "_le":return Number(b)<=Number(c);case "_lt":return Number(b)<Number(c);case "_re":var r;var u=a.ignore_case?"i":void 0;try{var p=String(c)+u,t=Gk.get(p);t||(t=new RegExp(c,u),Gk.set(p,t));r=t.test(b)}catch(v){r=!1}return r;case "_sw":return 0==String(b).indexOf(String(c));case "_um":return Hk(b,
c)}return!1};var Kk=function(a,b){var c=function(){};c.prototype=a.prototype;var d=new c;a.apply(d,Array.prototype.slice.call(arguments,1));return d};var Lk={},Mk=encodeURI,Y=encodeURIComponent,Nk=nc;var Ok=function(a,b){if(!a)return!1;var c=Ze($e(a),"host");if(!c)return!1;for(var d=0;b&&d<b.length;d++){var e=b[d]&&b[d].toLowerCase();if(e){var f=c.length-e.length;0<f&&"."!=e.charAt(0)&&(f--,e="."+e);if(0<=f&&c.indexOf(e,f)==f)return!0}}return!1};
var Pk=function(a,b,c){for(var d={},e=!1,f=0;a&&f<a.length;f++)a[f]&&a[f].hasOwnProperty(b)&&a[f].hasOwnProperty(c)&&(d[a[f][b]]=a[f][c],e=!0);return e?d:null};Lk.pg=function(){var a=!1;return a};var am=function(){var a=D.gaGlobal=D.gaGlobal||{};a.hid=a.hid||xa();return a.hid};var lm=window,mm=document,nm=function(a){var b=lm._gaUserPrefs;if(b&&b.ioo&&b.ioo()||a&&!0===lm["ga-disable-"+a])return!0;try{var c=lm.external;if(c&&c._gaUserPrefs&&"oo"==c._gaUserPrefs)return!0}catch(f){}for(var d=Ff("AMP_TOKEN",mm.cookie,!0),e=0;e<d.length;e++)if("$OPT_OUT"==d[e])return!0;return mm.getElementById("__gaOptOutExtension")?!0:!1};var qm=function(a){Aa(a,function(c){"_"===c.charAt(0)&&delete a[c]});var b=a[H.ba]||{};Aa(b,function(c){"_"===c.charAt(0)&&delete b[c]})};var um=function(a,b,c){ki(b,c,a)},vm=function(a,b,c){ki(b,c,a,!0)},xm=function(a,b){};
function wm(a,b){}var Z={a:{}};


Z.a.jsm=["customScripts"],function(){(function(a){Z.__jsm=a;Z.__jsm.b="jsm";Z.__jsm.g=!0;Z.__jsm.priorityOverride=0})(function(a){if(void 0!==a.vtp_javascript){var b=a.vtp_javascript;try{var c=X("google_tag_manager");return c&&c.e&&c.e(b)}catch(d){}}})}();

Z.a.c=["google"],function(){(function(a){Z.__c=a;Z.__c.b="c";Z.__c.g=!0;Z.__c.priorityOverride=0})(function(a){return a.vtp_value})}();
Z.a.e=["google"],function(){(function(a){Z.__e=a;Z.__e.b="e";Z.__e.g=!0;Z.__e.priorityOverride=0})(function(a){return String(Td(a.vtp_gtmEventId,"event"))})}();
Z.a.f=["google"],function(){(function(a){Z.__f=a;Z.__f.b="f";Z.__f.g=!0;Z.__f.priorityOverride=0})(function(a){var b=W("gtm.referrer",1)||E.referrer;return b?a.vtp_component&&"URL"!=a.vtp_component?Ze($e(String(b)),a.vtp_component,a.vtp_stripWww,a.vtp_defaultPages,a.vtp_queryKey):ak(String(b)):String(b)})}();
Z.a.cl=["google"],function(){function a(b){var c=b.target;if(c){var d=pj(c,"gtm.click");bk(d)}}(function(b){Z.__cl=b;Z.__cl.b="cl";Z.__cl.g=!0;Z.__cl.priorityOverride=0})(function(b){if(!fk("cl")){var c=X("document");oc(c,"click",a,!0);gk("cl")}G(b.vtp_gtmOnSuccess)})}();
Z.a.j=["google"],function(){(function(a){Z.__j=a;Z.__j.b="j";Z.__j.g=!0;Z.__j.priorityOverride=0})(function(a){for(var b=String(a.vtp_name).split("."),c=X(b.shift()),d=0;d<b.length;d++)c=c&&c[b[d]];return c})}();Z.a.k=["google"],function(){(function(a){Z.__k=a;Z.__k.b="k";Z.__k.g=!0;Z.__k.priorityOverride=0})(function(a){return dk(a.vtp_name,W("gtm.cookie",1),!!a.vtp_decodeCookie)[0]})}();

Z.a.u=["google"],function(){var a=function(b){return{toString:function(){return b}}};(function(b){Z.__u=b;Z.__u.b="u";Z.__u.g=!0;Z.__u.priorityOverride=0})(function(b){var c;b.vtp_customUrlSource?c=b.vtp_customUrlSource:c=W("gtm.url",1);c=c||Yj();var d=b[a("vtp_component")];if(!d||"URL"==d)return ak(String(c));var e=$e(String(c)),f;if("QUERY"===d)a:{var h=b[a("vtp_multiQueryKeys").toString()],k=b[a("vtp_queryKey").toString()]||"",l=b[a("vtp_ignoreEmptyQueryParam").toString()],m;h?va(k)?m=k:m=String(k).replace(/\s+/g,
"").split(","):m=[String(k)];for(var n=0;n<m.length;n++){var r=Ze(e,"QUERY",void 0,void 0,m[n]);if(void 0!=r&&(!l||""!==r)){f=r;break a}}f=void 0}else f=Ze(e,d,"HOST"==d?b[a("vtp_stripWww")]:void 0,"PATH"==d?b[a("vtp_defaultPages")]:void 0,void 0);return f})}();
Z.a.v=["google"],function(){(function(a){Z.__v=a;Z.__v.b="v";Z.__v.g=!0;Z.__v.priorityOverride=0})(function(a){var b=a.vtp_name;if(!b||!b.replace)return!1;var c=W(b.replace(/\\\./g,"."),a.vtp_dataLayerVersion||1);return void 0!==c?c:a.vtp_defaultValue})}();
Z.a.ua=["google"],function(){var a,b={},c=function(d){var e={},f={},h={},k={},l={},m=void 0;if(d.vtp_gaSettings){var n=d.vtp_gaSettings;C(Pk(n.vtp_fieldsToSet,"fieldName","value"),f);C(Pk(n.vtp_contentGroup,"index","group"),h);C(Pk(n.vtp_dimension,"index","dimension"),k);C(Pk(n.vtp_metric,"index","metric"),l);d.vtp_gaSettings=null;n.vtp_fieldsToSet=void 0;n.vtp_contentGroup=void 0;n.vtp_dimension=void 0;n.vtp_metric=void 0;var r=C(n);d=C(d,r)}C(Pk(d.vtp_fieldsToSet,"fieldName","value"),f);C(Pk(d.vtp_contentGroup,
"index","group"),h);C(Pk(d.vtp_dimension,"index","dimension"),k);C(Pk(d.vtp_metric,"index","metric"),l);var u=Pe(d.vtp_functionName);if(ra(u)){var p="",t="";d.vtp_setTrackerName&&"string"==typeof d.vtp_trackerName?""!==d.vtp_trackerName&&(t=d.vtp_trackerName,p=t+"."):(t="gtm"+hd(),p=t+".");var v={name:!0,clientId:!0,sampleRate:!0,siteSpeedSampleRate:!0,alwaysSendReferrer:!0,allowAnchor:!0,allowLinker:!0,cookieName:!0,cookieDomain:!0,cookieExpires:!0,cookiePath:!0,cookieUpdate:!0,legacyCookieDomain:!0,
legacyHistoryImport:!0,storage:!0,useAmpClientId:!0,storeGac:!0},w={allowAnchor:!0,allowLinker:!0,alwaysSendReferrer:!0,anonymizeIp:!0,cookieUpdate:!0,exFatal:!0,forceSSL:!0,javaEnabled:!0,legacyHistoryImport:!0,nonInteraction:!0,useAmpClientId:!0,useBeacon:!0,storeGac:!0,allowAdFeatures:!0,allowAdPersonalizationSignals:!0},y=function(O){var K=[].slice.call(arguments,0);K[0]=p+K[0];u.apply(window,K)},x=function(O,K){return void 0===K?K:O(K)},B=function(O,K){if(K)for(var ta in K)K.hasOwnProperty(ta)&&
y("set",O+ta,K[ta])},z=function(){},A=function(O,K,ta){var Hb=0;if(O)for(var Da in O)if(O.hasOwnProperty(Da)&&(ta&&v[Da]||!ta&&void 0===v[Da])){var Za=w[Da]?Ca(O[Da]):O[Da];"anonymizeIp"!=Da||Za||(Za=void 0);K[Da]=Za;Hb++}return Hb},F={name:t};A(f,F,
!0);u("create",d.vtp_trackingId||e.trackingId,F);y("set","&gtm",Ng(!0));d.vtp_enableRecaptcha&&y("require","recaptcha","recaptcha.js");(function(O,K){void 0!==d[K]&&y("set",O,d[K])})("nonInteraction","vtp_nonInteraction");B("contentGroup",h);B("dimension",k);B("metric",l);var J={};A(f,J,!1)&&y("set",J);var M;d.vtp_enableLinkId&&y("require","linkid","linkid.js");y("set","hitCallback",function(){var O=f&&f.hitCallback;ra(O)&&O();d.vtp_gtmOnSuccess()});if("TRACK_EVENT"==d.vtp_trackType){d.vtp_enableEcommerce&&(y("require","ec","ec.js"),z());var U={hitType:"event",eventCategory:String(d.vtp_eventCategory||e.category),eventAction:String(d.vtp_eventAction||
e.action),eventLabel:x(String,d.vtp_eventLabel||e.label),eventValue:x(Ba,d.vtp_eventValue||e.value)};A(M,U,!1);y("send",U);}else if("TRACK_SOCIAL"==d.vtp_trackType){}else if("TRACK_TRANSACTION"==
d.vtp_trackType){}else if("TRACK_TIMING"==d.vtp_trackType){}else if("DECORATE_LINK"==
d.vtp_trackType){}else if("DECORATE_FORM"==d.vtp_trackType){}else if("TRACK_DATA"==d.vtp_trackType){}else{d.vtp_enableEcommerce&&
(y("require","ec","ec.js"),z());if(d.vtp_doubleClick||"DISPLAY_FEATURES"==d.vtp_advertisingFeaturesType){var pa="_dc_gtm_"+String(d.vtp_trackingId).replace(/[^A-Za-z0-9-]/g,"");y("require","displayfeatures",void 0,{cookieName:pa})}if("DISPLAY_FEATURES_WITH_REMARKETING_LISTS"==d.vtp_advertisingFeaturesType){var ma="_dc_gtm_"+String(d.vtp_trackingId).replace(/[^A-Za-z0-9-]/g,"");y("require","adfeatures",{cookieName:ma})}M?y("send","pageview",M):y("send","pageview");}if(!a){var ua=d.vtp_useDebugVersion?"u/analytics_debug.js":"analytics.js";d.vtp_useInternalVersion&&!d.vtp_useDebugVersion&&(ua="internal/"+ua);a=!0;var ab=R("https:","http:","//www.google-analytics.com/"+ua,f&&f.forceSSL);
V(ab,function(){var O=Ne();O&&O.loaded||d.vtp_gtmOnFailure();},d.vtp_gtmOnFailure)}}else G(d.vtp_gtmOnFailure)};Z.__ua=c;Z.__ua.b="ua";Z.__ua.g=!0;Z.__ua.priorityOverride=0}();





Z.a.gclidw=["google"],function(){var a=["aw","dc","gf","ha","gp"];(function(b){Z.__gclidw=b;Z.__gclidw.b="gclidw";Z.__gclidw.g=!0;Z.__gclidw.priorityOverride=100})(function(b){G(b.vtp_gtmOnSuccess);var c,d,e;b.vtp_enableCookieOverrides&&(e=b.vtp_cookiePrefix,c=b.vtp_path,d=b.vtp_domain);var f=null;b.vtp_enableCookieUpdateFeature&&(f=!0,void 0!==b.vtp_cookieUpdate&&(f=!!b.vtp_cookieUpdate));var h=e,k=c,l=d;if(b.vtp_enableCrossDomainFeature&&(!b.vtp_enableCrossDomain||!1!==b.vtp_acceptIncoming)&&(b.vtp_enableCrossDomain||
qg())){Eg(a,h,k,l,void 0);}var m={prefix:e,path:c,domain:d,Ka:void 0};Bg(m);Ig(["aw","dc"],m);mh(f,e,c,d);var n=e;if(b.vtp_enableCrossDomainFeature&&b.vtp_enableCrossDomain&&b.vtp_linkerDomains){var r=b.vtp_linkerDomains.toString().replace(/\s+/g,"").split(","),u=b.vtp_urlPosition,p=!!b.vtp_formDecoration;Gg(a,r,u,p,n);}})}();

Z.a.aev=["google"],function(){function a(p,t){var v=Td(p,"gtm");if(v)return v[t]}function b(p,t,v,w){w||(w="element");var y=p+"."+t,x;if(n.hasOwnProperty(y))x=n[y];else{var B=a(p,w);if(B&&(x=v(B),n[y]=x,r.push(y),35<r.length)){var z=r.shift();delete n[z]}}return x}function c(p,t,v){var w=a(p,u[t]);return void 0!==w?w:v}function d(p,t){if(!p)return!1;var v=e(Yj());va(t)||(t=String(t||"").replace(/\s+/g,"").split(","));for(var w=[v],y=0;y<t.length;y++)if(t[y]instanceof RegExp){if(t[y].test(p))return!1}else{var x=
t[y];if(0!=x.length){if(0<=e(p).indexOf(x))return!1;w.push(e(x))}}return!Ok(p,w)}function e(p){m.test(p)||(p="http://"+p);return Ze($e(p),"HOST",!0)}function f(p,t,v){switch(p){case "SUBMIT_TEXT":return b(t,"FORM."+p,h,"formSubmitElement")||v;case "LENGTH":var w=b(t,"FORM."+p,k);return void 0===w?v:w;case "INTERACTED_FIELD_ID":return l(t,"id",v);case "INTERACTED_FIELD_NAME":return l(t,"name",v);case "INTERACTED_FIELD_TYPE":return l(t,"type",v);case "INTERACTED_FIELD_POSITION":var y=a(t,"interactedFormFieldPosition");
return void 0===y?v:y;case "INTERACT_SEQUENCE_NUMBER":var x=a(t,"interactSequenceNumber");return void 0===x?v:x;default:return v}}function h(p){switch(p.tagName.toLowerCase()){case "input":return qc(p,"value");case "button":return sc(p);default:return null}}function k(p){if("form"===p.tagName.toLowerCase()&&p.elements){for(var t=0,v=0;v<p.elements.length;v++)wj(p.elements[v])&&t++;return t}}function l(p,t,v){var w=a(p,"interactedFormField");return w&&qc(w,t)||v}var m=/^https?:\/\//i,n={},r=[],u={ATTRIBUTE:"elementAttribute",
CLASSES:"elementClasses",ELEMENT:"element",ID:"elementId",HISTORY_CHANGE_SOURCE:"historyChangeSource",HISTORY_NEW_STATE:"newHistoryState",HISTORY_NEW_URL_FRAGMENT:"newUrlFragment",HISTORY_OLD_STATE:"oldHistoryState",HISTORY_OLD_URL_FRAGMENT:"oldUrlFragment",TARGET:"elementTarget"};(function(p){Z.__aev=p;Z.__aev.b="aev";Z.__aev.g=!0;Z.__aev.priorityOverride=0})(function(p){var t=p.vtp_gtmEventId,v=p.vtp_defaultValue,w=p.vtp_varType;switch(w){case "TAG_NAME":var y=a(t,"element");return y&&y.tagName||
v;case "TEXT":return b(t,w,sc)||v;case "URL":var x;a:{var B=String(a(t,"elementUrl")||v||""),z=$e(B),A=String(p.vtp_component||"URL");switch(A){case "URL":x=B;break a;case "IS_OUTBOUND":x=d(B,p.vtp_affiliatedDomains);break a;default:x=Ze(z,A,p.vtp_stripWww,p.vtp_defaultPages,p.vtp_queryKey)}}return x;case "ATTRIBUTE":var F;if(void 0===p.vtp_attribute)F=c(t,w,v);else{var J=p.vtp_attribute,M=a(t,"element");F=M&&qc(M,J)||v||""}return F;case "MD":var U=p.vtp_mdValue,fa=b(t,"MD",Ij);return U&&fa?Lj(fa,
U)||v:fa||v;case "FORM":return f(String(p.vtp_component||"SUBMIT_TEXT"),t,v);default:return c(t,w,v)}})}();

Z.a.cegg=["nonGoogleScripts"],function(){var a={};(function(b){Z.__cegg=b;Z.__cegg.b="cegg";Z.__cegg.g=!0;Z.__cegg.priorityOverride=0})(function(b){try{var c=b.vtp_usersNumericId;if(c)if(a.hasOwnProperty(c)&&!0===a[c])b.vtp_gtmOnSuccess();else{b.vtp_snapshotName&&(D.CE_SNAPSHOT_NAME=b.vtp_snapshotName);for(var d=c.toString();8>d.length;)d="0"+d;var e=d.replace(/(\d+)(\d{4})$/,"/pages/scripts/$1/$2.js");V("//script.crazyegg.com"+e+"?"+Math.floor((new Date).getTime()/36E5),b.vtp_gtmOnSuccess,b.vtp_gtmOnFailure);
a[c]=!0}else G(b.vtp_gtmOnFailure)}catch(f){G(b.vtp_gtmOnFailure)}})}();
Z.a.awct=["google"],function(){var a=!1,b=[],c=function(k){var l=X("google_trackConversion"),m=k.gtm_onFailure;"function"==typeof l?l(k)||m():m()},d=function(){for(;0<b.length;)c(b.shift())},e=function(){return function(){d();a=!1}},f=function(){return function(){d();b={push:c};}},h=function(k){Ie();var l={google_basket_transaction_type:"purchase",google_conversion_domain:"",google_conversion_id:k.vtp_conversionId,google_conversion_label:k.vtp_conversionLabel,
google_conversion_value:k.vtp_conversionValue||0,google_remarketing_only:!1,onload_callback:k.vtp_gtmOnSuccess,gtm_onFailure:k.vtp_gtmOnFailure,google_gtm:Ng()};k.vtp_rdp&&(l.google_restricted_data_processing=!0);var m=function(v){return function(w,y,x){var B="DATA_LAYER"==v?W(x):k[y];B&&(l[w]=B)}},n=m("JSON");n("google_conversion_currency","vtp_currencyCode");n("google_conversion_order_id","vtp_orderId");k.vtp_enableProductReporting&&(n=m(k.vtp_productReportingDataSource),n("google_conversion_merchant_id",
"vtp_awMerchantId","aw_merchant_id"),n("google_basket_feed_country","vtp_awFeedCountry","aw_feed_country"),n("google_basket_feed_language","vtp_awFeedLanguage","aw_feed_language"),n("google_basket_discount","vtp_discount","discount"),n("google_conversion_items","vtp_items","items"),l.google_conversion_items&&l.google_conversion_items.map&&(l.google_conversion_items=l.google_conversion_items.map(function(v){return{value:v.price,quantity:v.quantity,item_id:v.id}})));var r=function(v,w){(l.google_additional_conversion_params=
l.google_additional_conversion_params||{})[v]=w},u=function(v){return function(w,y,x,B){var z="DATA_LAYER"==v?W(x):k[y];B(z)&&r(w,z)}},p=-1==navigator.userAgent.toLowerCase().indexOf("firefox")?"//www.googleadservices.com/pagead/conversion_async.js":"https://www.google.com/pagead/conversion_async.js";k.vtp_enableNewCustomerReporting&&(n=u(k.vtp_newCustomerReportingDataSource),n("vdnc","vtp_awNewCustomer","new_customer",function(v){return void 0!=v&&""!==v}),n("vdltv","vtp_awCustomerLTV","customer_lifetime_value",
function(v){return void 0!=v&&""!==v}));!k.hasOwnProperty("vtp_enableConversionLinker")||k.vtp_enableConversionLinker?(k.vtp_conversionCookiePrefix&&(l.google_gcl_cookie_prefix=k.vtp_conversionCookiePrefix),l.google_read_gcl_cookie_opt_out=!1):l.google_read_gcl_cookie_opt_out=!0;var t=!0;t&&b.push(l);a||(a=!0,
V(p,f(),e(p)))};Z.__awct=h;Z.__awct.b="awct";Z.__awct.g=!0;Z.__awct.priorityOverride=0}();Z.a.remm=["google"],function(){(function(a){Z.__remm=a;Z.__remm.b="remm";Z.__remm.g=!0;Z.__remm.priorityOverride=0})(function(a){for(var b=a.vtp_input,c=a.vtp_map||[],d=a.vtp_fullMatch,e=a.vtp_ignoreCase?"gi":"g",f=0;f<c.length;f++){var h=c[f].key||"";d&&(h="^"+h+"$");var k=new RegExp(h,e);if(k.test(b)){var l=c[f].value;a.vtp_replaceAfterMatch&&(l=String(b).replace(k,l));return l}}return a.vtp_defaultValue})}();

Z.a.fsl=[],function(){function a(){var e=X("document"),f=c(),h=HTMLFormElement.prototype.submit;oc(e,"click",function(k){var l=k.target;if(l&&(l=uc(l,["button","input"],100))&&("submit"==l.type||"image"==l.type)&&l.name&&qc(l,"value")){var m;l.form?l.form.tagName?m=l.form:m=E.getElementById(l.form):m=uc(l,["form"],100);m&&f.store(m,l)}},!1);oc(e,"submit",function(k){var l=k.target;if(!l)return k.returnValue;var m=k.defaultPrevented||!1===k.returnValue,n=b(l)&&!m,r=f.get(l),u=!0;if(d(l,function(){if(u){var p;
r&&(p=e.createElement("input"),p.type="hidden",p.name=r.name,p.value=r.value,l.appendChild(p));h.call(l);p&&l.removeChild(p)}},m,n,r))u=!1;else return m||(k.preventDefault&&k.preventDefault(),k.returnValue=!1),!1;return k.returnValue},!1);HTMLFormElement.prototype.submit=function(){var k=this,l=b(k),m=!0;d(k,function(){m&&h.call(k)},!1,l)&&(h.call(k),m=!1)}}function b(e){var f=e.target;return f&&"_self"!==f&&"_parent"!==f&&"_top"!==f?!1:!0}function c(){var e=[],f=function(h){return wa(e,function(k){return k.form===
h})};return{store:function(h,k){var l=f(h);l?l.button=k:e.push({form:h,button:k})},get:function(h){var k=f(h);return k?k.button:null}}}function d(e,f,h,k,l){var m=tj("fsl",h?"nv.mwt":"mwt",0),n;n=h?tj("fsl","nv.ids",[]):tj("fsl","ids",[]);if(!n.length)return!0;var r=pj(e,"gtm.formSubmit",n),u=e.action;u&&u.tagName&&(u=e.cloneNode(!1).action);r["gtm.elementUrl"]=u;l&&(r["gtm.formSubmitElement"]=l);if(k&&m){if(!bk(r,fj(f),m))return!1}else bk(r,function(){},m||2E3);return!0}(function(e){Z.__fsl=e;Z.__fsl.b=
"fsl";Z.__fsl.g=!0;Z.__fsl.priorityOverride=0})(function(e){var f=e.vtp_waitForTags,h=e.vtp_checkValidation,k=Number(e.vtp_waitForTagsTimeout);if(!k||0>=k)k=2E3;var l=e.vtp_uniqueTriggerId||"0";if(f){var m=function(r){return Math.max(k,r)};sj("fsl","mwt",m,0);h||sj("fsl","nv.mwt",m,0)}var n=function(r){r.push(l);return r};sj("fsl","ids",n,[]);h||sj("fsl","nv.ids",n,[]);fk("fsl")||(a(),gk("fsl"));G(e.vtp_gtmOnSuccess)})}();




Z.a.mf=["nonGoogleScripts"],function(){var a={},b=function(c){try{c.vtp_path&&(D.mouseflowPath=c.vtp_path);var d=Number(c.vtp_htmlDelay);0<d&&(D.mouseflowHtmlDelay=d);var e=c.vtp_customVars;if(e){var f=X("_mfq",[],!0),h;for(h in e)e.hasOwnProperty(h)&&f.push(["setVariable",h,e[h]])}a.hasOwnProperty(c.vtp_projectId)&&!0===a[c.vtp_projectId]||(V("//cdn.mouseflow.com/projects/"+c.vtp_projectId+".js",c.vtp_gtmOnSuccess,c.vtp_gtmOnFailure),a[c.vtp_projectId]=!0)}catch(k){G(c.vtp_gtmOnFailure)}};Z.__mf=b;Z.__mf.b="mf";Z.__mf.g=!0;Z.__mf.priorityOverride=0}();
Z.a.html=["customScripts"],function(){function a(d,e,f,h){return function(){try{if(0<e.length){var k=e.shift(),l=a(d,e,f,h);if("SCRIPT"==String(k.nodeName).toUpperCase()&&"text/gtmscript"==k.type){var m=E.createElement("script");m.async=!1;m.type="text/javascript";m.id=k.id;m.text=k.text||k.textContent||k.innerHTML||"";k.charset&&(m.charset=k.charset);var n=k.getAttribute("data-gtmsrc");n&&(m.src=n,jc(m,l));d.insertBefore(m,null);n||l()}else if(k.innerHTML&&0<=k.innerHTML.toLowerCase().indexOf("<script")){for(var r=
[];k.firstChild;)r.push(k.removeChild(k.firstChild));d.insertBefore(k,null);a(k,r,l,h)()}else d.insertBefore(k,null),l()}else f()}catch(u){G(h)}}}var c=function(d){if(E.body){var e=
d.vtp_gtmOnFailure,f=jk(d.vtp_html,d.vtp_gtmOnSuccess,e),h=f.Jc,k=f.B;if(d.vtp_useIframe){}else d.vtp_supportDocumentWrite?b(h,k,e):a(E.body,tc(h),k,e)()}else Xj(function(){c(d)},
200)};Z.__html=c;Z.__html.b="html";Z.__html.g=!0;Z.__html.priorityOverride=0}();






Z.a.lcl=[],function(){function a(){var c=X("document"),d=0,e=function(f){var h=f.target;if(h&&3!==f.which&&!(f.ng||f.timeStamp&&f.timeStamp===d)){d=f.timeStamp;h=uc(h,["a","area"],100);if(!h)return f.returnValue;var k=f.defaultPrevented||!1===f.returnValue,l=tj("lcl",k?"nv.mwt":"mwt",0),m;m=k?tj("lcl","nv.ids",[]):tj("lcl","ids",[]);if(m.length){var n=pj(h,"gtm.linkClick",m);if(b(f,h,c)&&!k&&l&&h.href){var r=String(ik(h,"rel")||""),u=!!wa(r.split(" "),function(v){return"noreferrer"===v.toLowerCase()});
u&&P("GTM",36);var p=X((ik(h,"target")||"_self").substring(1)),t=!0;if(bk(n,fj(function(){var v;if(v=t&&p){var w;a:if(u){var y;try{y=new MouseEvent(f.type)}catch(x){if(!c.createEvent){w=!1;break a}y=c.createEvent("MouseEvents");y.initEvent(f.type,!0,!0)}y.ng=!0;f.target.dispatchEvent(y);w=!0}else w=!1;v=!w}v&&(p.location.href=ik(h,"href"))}),l))t=!1;else return f.preventDefault&&f.preventDefault(),f.returnValue=!1}else bk(n,function(){},l||2E3);return!0}}};oc(c,"click",e,!1);oc(c,"auxclick",e,!1)}
function b(c,d,e){if(2===c.which||c.ctrlKey||c.shiftKey||c.altKey||c.metaKey)return!1;var f=ik(d,"href"),h=f.indexOf("#"),k=ik(d,"target");if(k&&"_self"!==k&&"_parent"!==k&&"_top"!==k||0===h)return!1;if(0<h){var l=ak(f),m=ak(e.location);return l!==m}return!0}(function(c){Z.__lcl=c;Z.__lcl.b="lcl";Z.__lcl.g=!0;Z.__lcl.priorityOverride=0})(function(c){var d=void 0===c.vtp_waitForTags?!0:c.vtp_waitForTags,e=void 0===c.vtp_checkValidation?!0:c.vtp_checkValidation,f=Number(c.vtp_waitForTagsTimeout);if(!f||
0>=f)f=2E3;var h=c.vtp_uniqueTriggerId||"0";if(d){var k=function(m){return Math.max(f,m)};sj("lcl","mwt",k,0);e||sj("lcl","nv.mwt",k,0)}var l=function(m){m.push(h);return m};sj("lcl","ids",l,[]);e||sj("lcl","nv.ids",l,[]);fk("lcl")||(a(),gk("lcl"));G(c.vtp_gtmOnSuccess)})}();

var ym={};ym.macro=function(a){if(mj.xc.hasOwnProperty(a))return mj.xc[a]},ym.onHtmlSuccess=mj.ce(!0),ym.onHtmlFailure=mj.ce(!1);ym.dataLayer=Md;ym.callback=function(a){fd.hasOwnProperty(a)&&ra(fd[a])&&fd[a]();delete fd[a]};function zm(){I[Wc.s]=ym;Ja(gd,Z.a);zb=zb||mj;Ab=qe}
function Am(){sg.gtm_3pds=!0;I=D.google_tag_manager=D.google_tag_manager||{};if(I[Wc.s]){var a=I.zones;a&&a.unregisterChild(Wc.s)}else{for(var b=data.resource||{},c=b.macros||[],d=0;d<c.length;d++)rb.push(c[d]);for(var e=b.tags||[],f=0;f<e.length;f++)ub.push(e[f]);for(var h=b.predicates||[],k=0;k<
h.length;k++)tb.push(h[k]);for(var l=b.rules||[],m=0;m<l.length;m++){for(var n=l[m],r={},u=0;u<n.length;u++)r[n[u][0]]=Array.prototype.slice.call(n[u],1);sb.push(r)}xb=Z;yb=Ik;zm();lj();ue=!1;ve=0;if("interactive"==E.readyState&&!E.createEventObject||"complete"==E.readyState)xe();else{oc(E,"DOMContentLoaded",xe);oc(E,"readystatechange",xe);if(E.createEventObject&&E.documentElement.doScroll){var p=!0;try{p=!D.frameElement}catch(y){}p&&ye()}oc(D,"load",xe)}Ji=!1;"complete"===E.readyState?Li():oc(D,
"load",Li);a:{if(!Bd)break a;D.setInterval(Cd,864E5);}
cd=(new Date).getTime();
}}Am();

})()
