/**
 * DataTable - Sortable, filterable table with pagination and CSV export.
 *
 * Usage:
 *   const table = new DataTable({
 *     containerId: 'my-table-container',
 *     columns: [
 *       { key: 'date', label: 'Date', sortable: true },
 *       { key: 'sku', label: 'SKU', sortable: true, filterable: true },
 *       { key: 'judgment', label: 'Judgment', formatter: (v) => `<span class="badge">${v}</span>` },
 *     ],
 *     fetchData: async (params) => { rows, total },
 *     pageSize: 20,
 *   });
 *   table.load();
 */

export class DataTable {
    /**
     * @param {Object} options
     * @param {string} options.containerId
     * @param {Array<{key:string, label:string, sortable?:boolean, formatter?:Function}>} options.columns
     * @param {Function} options.fetchData - async (params) => {rows: Array, total: number}
     * @param {number} [options.pageSize=20]
     * @param {string} [options.emptyMessage='No data available']
     */
    constructor(options) {
        this.container = document.getElementById(options.containerId);
        this.columns = options.columns;
        this.fetchData = options.fetchData;
        this.pageSize = options.pageSize || 20;
        this.emptyMessage = options.emptyMessage || 'No data available';

        this.currentPage = 1;
        this.totalRows = 0;
        this.sortKey = null;
        this.sortDir = 'asc';
        this.filters = {};
        this._rows = [];

        if (this.container) {
            this._render();
        }
    }

    _render() {
        this.container.innerHTML = '';

        // Table
        const table = document.createElement('table');
        table.className = 'data-table';
        table.setAttribute('role', 'grid');
        table.setAttribute('aria-label', 'Data table');

        // Thead
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headerRow.setAttribute('role', 'row');

        this.columns.forEach(col => {
            const th = document.createElement('th');
            th.setAttribute('role', 'columnheader');
            th.setAttribute('scope', 'col');
            th.textContent = col.label;

            if (col.sortable) {
                th.classList.add('sortable');
                th.setAttribute('tabindex', '0');
                th.setAttribute('aria-sort', 'none');
                const icon = document.createElement('i');
                icon.className = 'fa-solid fa-sort sort-icon';
                icon.setAttribute('aria-hidden', 'true');
                th.appendChild(icon);

                th.addEventListener('click', () => this._onSort(col.key, th));
                th.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        this._onSort(col.key, th);
                    }
                });
            }

            headerRow.appendChild(th);
        });

        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Tbody
        this._tbody = document.createElement('tbody');
        this._tbody.setAttribute('role', 'rowgroup');
        this._tbody.setAttribute('aria-live', 'polite');
        table.appendChild(this._tbody);

        this.container.appendChild(table);

        // Pagination bar
        this._paginationBar = document.createElement('div');
        this._paginationBar.className = 'data-table-pagination';
        this.container.appendChild(this._paginationBar);
    }

    _onSort(key, thElement) {
        if (this.sortKey === key) {
            this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortKey = key;
            this.sortDir = 'asc';
        }

        // Update aria-sort on all headers
        this.container.querySelectorAll('th[role="columnheader"]').forEach(th => {
            th.setAttribute('aria-sort', 'none');
            const icon = th.querySelector('.sort-icon');
            if (icon) icon.className = 'fa-solid fa-sort sort-icon';
        });

        thElement.setAttribute('aria-sort', this.sortDir === 'asc' ? 'ascending' : 'descending');
        const icon = thElement.querySelector('.sort-icon');
        if (icon) {
            icon.className = this.sortDir === 'asc'
                ? 'fa-solid fa-sort-up sort-icon'
                : 'fa-solid fa-sort-down sort-icon';
        }

        this.load(1);
    }

    _renderRows(rows) {
        this._tbody.innerHTML = '';

        if (!rows || rows.length === 0) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = this.columns.length;
            td.className = 'data-table-empty';
            td.textContent = this.emptyMessage;
            tr.appendChild(td);
            this._tbody.appendChild(tr);
            return;
        }

        rows.forEach(row => {
            const tr = document.createElement('tr');
            tr.setAttribute('role', 'row');

            this.columns.forEach(col => {
                const td = document.createElement('td');
                const value = row[col.key];
                if (col.formatter) {
                    td.innerHTML = col.formatter(value, row);
                } else {
                    td.textContent = value ?? '';
                }
                tr.appendChild(td);
            });

            this._tbody.appendChild(tr);
        });
    }

    _renderPagination() {
        const totalPages = Math.max(1, Math.ceil(this.totalRows / this.pageSize));
        this._paginationBar.innerHTML = '';

        const info = document.createElement('span');
        info.className = 'data-table-page-info';
        info.textContent = `Page ${this.currentPage} of ${totalPages} (${this.totalRows} records)`;

        const btnPrev = document.createElement('button');
        btnPrev.className = 'btn btn-secondary btn-sm';
        btnPrev.textContent = 'Prev';
        btnPrev.disabled = this.currentPage <= 1;
        btnPrev.setAttribute('aria-label', 'Previous page');
        btnPrev.addEventListener('click', () => this.load(this.currentPage - 1));

        const btnNext = document.createElement('button');
        btnNext.className = 'btn btn-secondary btn-sm';
        btnNext.textContent = 'Next';
        btnNext.disabled = this.currentPage >= totalPages;
        btnNext.setAttribute('aria-label', 'Next page');
        btnNext.addEventListener('click', () => this.load(this.currentPage + 1));

        this._paginationBar.appendChild(btnPrev);
        this._paginationBar.appendChild(info);
        this._paginationBar.appendChild(btnNext);
    }

    /**
     * Load data for a given page.
     * @param {number} [page=1]
     */
    async load(page = 1) {
        this.currentPage = page;
        const params = {
            skip: (page - 1) * this.pageSize,
            limit: this.pageSize,
            sort_key: this.sortKey,
            sort_dir: this.sortDir,
            ...this.filters,
        };

        try {
            const result = await this.fetchData(params);
            this._rows = result.rows || [];
            this.totalRows = result.total || 0;
            this._renderRows(this._rows);
            this._renderPagination();
        } catch (err) {
            console.error('DataTable load error:', err);
            this._renderRows([]);
            this._renderPagination();
        }
    }

    /**
     * Apply filters and reload.
     * @param {Object} filters - key/value pairs
     */
    filter(filters) {
        this.filters = { ...filters };
        this.load(1);
    }

    /**
     * Export current filter results as CSV download.
     * @param {string} [filename='export.csv']
     */
    async exportCsv(filename = 'export.csv') {
        try {
            const result = await this.fetchData({
                skip: 0,
                limit: 10000,
                sort_key: this.sortKey,
                sort_dir: this.sortDir,
                ...this.filters,
            });

            const rows = result.rows || [];
            if (rows.length === 0) return;

            const headers = this.columns.map(c => c.label);
            const csvRows = [headers.join(',')];

            rows.forEach(row => {
                const values = this.columns.map(col => {
                    const v = row[col.key];
                    const str = String(v ?? '').replace(/"/g, '""');
                    return `"${str}"`;
                });
                csvRows.push(values.join(','));
            });

            const blob = new Blob(['\uFEFF' + csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('CSV export error:', err);
        }
    }

    destroy() {
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}
