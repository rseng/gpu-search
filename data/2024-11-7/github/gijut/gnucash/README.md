# https://github.com/gijut/gnucash

```console
po/tr.po:msgstr " %s adresindeki sunucuda bir hata oluştu, veya hatalı ya da bozuk veriyle karşılaştı."
src/gnome-utils/gnc-tree-view-account.c:#define ACCT_COUNT    "NumberOfOpenAccounts"
src/gnome-utils/gnc-tree-view-account.c:#define ACCT_OPEN     "OpenAccount%d"
src/gnome-utils/ui/osx_accel_map:; (gtk_accel_path "<Actions>/GncPluginPageAccountTreeActions/FileOpenAccountAction" "")
src/gnc/mainwindow.ui:    <addaction name="actionOpenAccount"/>
src/gnc/mainwindow.ui:  <action name="actionOpenAccount">
src/gnome/window-reconcile2.c:            gtk_action_group_get_action (action_group, "AccountOpenAccountAction");
src/gnome/window-reconcile2.c:        "AccountOpenAccountAction", GTK_STOCK_JUMP_TO, N_("_Open Account"), NULL,
src/gnome/gnc-plugin-page-budget.c:        "OpenAccountAction", GNC_STOCK_OPEN_ACCOUNT, N_("Open _Account"), NULL,
src/gnome/gnc-plugin-page-budget.c:    "OpenAccountAction",
src/gnome/gnc-plugin-page-budget.c:    { "OpenAccountAction", 	    N_("Open") },
src/gnome/window-reconcile.c:            gtk_action_group_get_action (action_group, "AccountOpenAccountAction");
src/gnome/window-reconcile.c:        "AccountOpenAccountAction", GTK_STOCK_JUMP_TO, N_("_Open Account"), NULL,
src/gnome/ui/gnc-plugin-page-account-tree-ui.xml:        <menuitem name="FileOpenAccount" action="FileOpenAccountAction"/>
src/gnome/ui/gnc-plugin-page-account-tree-ui.xml:      <menuitem name="AccountOpenAccount" action="FileOpenAccountAction"/>
src/gnome/ui/gnc-plugin-page-account-tree-ui.xml:      <toolitem name="ToolbarOpenAccount" action="FileOpenAccountAction"/>
src/gnome/ui/gnc-reconcile-window-ui.xml:      <menuitem name="AccountOpenAccount" action="AccountOpenAccountAction"/>
src/gnome/ui/gnc-reconcile-window-ui.xml:    <toolitem name="AccountOpenAccount" action="AccountOpenAccountAction"/>
src/gnome/ui/gnc-plugin-page-account-tree2-ui.xml:        <menuitem name="FileOpenAccount2" action="FileOpenAccount2Action"/>
src/gnome/ui/gnc-plugin-page-account-tree2-ui.xml:        <menuitem name="FileOpenAccount" action="FileOpenAccountAction"/>
src/gnome/ui/gnc-plugin-page-account-tree2-ui.xml:      <menuitem name="AccountOpenAccount2" action="FileOpenAccount2Action"/>
src/gnome/ui/gnc-plugin-page-account-tree2-ui.xml:      <menuitem name="AccountOpenAccount" action="FileOpenAccountAction"/>
src/gnome/ui/gnc-plugin-page-account-tree2-ui.xml:      <toolitem name="ToolbarOpenAccount" action="FileOpenAccountAction"/>
src/gnome/ui/gnc-plugin-page-account-tree2-ui.xml:      <toolitem name="ToolbarOpenAccount2" action="FileOpenAccount2Action"/>
src/gnome/ui/gnc-plugin-page-budget-ui.xml:      <toolitem name="OpenAccount" action="OpenAccountAction"/>
src/gnome/gnc-plugin-page-account-tree.c:        "FileOpenAccount2Action", GNC_STOCK_OPEN_ACCOUNT, N_("Open _Account"), NULL,
src/gnome/gnc-plugin-page-account-tree.c:        "FileOpenAccountAction", GNC_STOCK_OPEN_ACCOUNT, N_("Open _Old Style Register Account"), NULL,
src/gnome/gnc-plugin-page-account-tree.c:        "FileOpenAccountAction", GNC_STOCK_OPEN_ACCOUNT, N_("Open _Account"), NULL,
src/gnome/gnc-plugin-page-account-tree.c:    "FileOpenAccountAction",
src/gnome/gnc-plugin-page-account-tree.c:    "FileOpenAccount2Action",
src/gnome/gnc-plugin-page-account-tree.c:    { "FileOpenAccountAction", 	            N_("Open") },
src/gnome/gnc-plugin-page-account-tree.c:    { "FileOpenAccount2Action", 	    N_("Open2") },

```
