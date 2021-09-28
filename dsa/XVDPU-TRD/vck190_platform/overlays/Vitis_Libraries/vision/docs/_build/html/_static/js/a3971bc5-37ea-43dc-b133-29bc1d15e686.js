window._mfq = window._mfq || [];
(function() {
    function whenThen(conditionFunc, intervalInMs, successFunc) {
        var rounds = 0;var timer = setInterval(function() {
            rounds++; 
            if (conditionFunc()) { 
                clearInterval(timer); 
                successFunc(); 
            }
            if (rounds > 3) { 
                clearInterval(timer); 
            }
        }, intervalInMs);
    }
    function googleAnalyticsIntegration() { 
        ga('create', 'UA-11440126-15'); 
        ga('set', 'dimension4', mouseflow.getSessionId()); 
        ga('send', 'event', 'Mouseflow', mouseflow.getSessionId(), {'nonInteraction': 1});
    }; 
    whenThen(function() { return window.ga && window.mouseflow; }, 1000, googleAnalyticsIntegration);
})();var mouseflowDisableKeyLogging = true;if(typeof mouseflow==='undefined'&&typeof mouseflowPlayback==='undefined'){(function(_1){function _0(){return undefined}function _6(){return null}function _5(){return false}function _7(_2){if(_2&&_2.length){for(var _4=0;_4<_2.length;_4++){this.push(_2[_4])}}};_7.prototype.push=function(_3){if(_3&&typeof _3==='function'){_3(mouseflow)}};_1.setTimeout(function(){if(!_1._mfq)_1._mfq=[];_1._mfq=new _7(_mfq)},1);_1.mouseflow={config:_0,start:_0,stop:_0,newPageView:_0,stopSession:_0,rebindEventHandlers:_0,getSessionId:_6,getPageViewId:_6,tag:_0,star:_0,setVariable:_0,identify:_0,formSubmitAttempt:_0,formSubmitSuccess:_0,formSubmitFailure:_0,debug:_0,isRecording:_5,isReturningUser:_5,activateFeedback:_0,websiteId:null,recordingRate:null,version:null}})(window)}