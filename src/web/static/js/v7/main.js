(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });

if (typeof v7.actions.initInspection === "function") v7.actions.initInspection();
if (typeof v7.actions.initRegistration === "function") v7.actions.initRegistration();
if (typeof v7.actions.initStdAdmin === "function") v7.actions.initStdAdmin();
if (typeof v7.actions.initResultTabs === "function") v7.actions.initResultTabs();
if (typeof v7.actions.initTestTab === "function") v7.actions.initTestTab();

if (typeof v7.products.loadProducts === "function") v7.products.loadProducts();
const refreshBtn = v7.utils.byId("btnRefreshProducts");
if (refreshBtn) refreshBtn.addEventListener("click", v7.products.loadProducts);

})();
