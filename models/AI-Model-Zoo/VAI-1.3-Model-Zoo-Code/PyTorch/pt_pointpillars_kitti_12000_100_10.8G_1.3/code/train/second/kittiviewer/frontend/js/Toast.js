var Toast = function(toasts, timeout = 3000){
    this.toasts = toasts;
    this.type = ['error', 'message', 'warning', 'success'];
    this.timeout = timeout;
}

Toast.prototype = {
    _addToast : function(type, text){
        var toast;
        toast = document.createElement('li');
        toast.classList.add(type);
        setTimeout(function(){
          toast.remove();
        }, this.timeout);
        this.toasts.appendChild(toast);
        return toast.innerHTML = `${type}: ${text}`;
    },

    message : function(text){
        return this._addToast("message", text);
    },
    warning : function(text){
        return this._addToast("warning", text);
    },
    error : function(text){
        return this._addToast("error", text);
    },
    success : function(text){
        return this._addToast("success", text);
    }
}
