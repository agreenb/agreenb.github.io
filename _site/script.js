$(document).ready(function() {
    $(document).on('hidden.bs.collapse', '.collapsible-table', function(){
    	$($(this).parent().find('.panel-heading').get(0)).addClass('collapsed')
    });

    $(document).on('show.bs.collapse', '.collapsible-table', function(){
        $($(this).parent().find('.panel-heading').get(0)).removeClass('collapsed')
    });
});