# Bản chỉnh train-only — ghi chú cho nhóm review

Đã kết hợp các gợi ý trong ba ảnh. Mẫu bố cục vẫn là Stock_META_Thien. Chưa thực hiện git init, commit, tạo remote hay push.

## Phần 1 được sửa như thế nào

| Mục | Thay đổi |
|---|---|
| 1.1 Load/split | Đọc date/time để chia theo thời gian trước; bảng split chỉ có n/start/end. Giữ đúng membership của cách chia cũ, kể cả giao dịch cùng timestamp. |
| 1.2 Sanity | Đọc đủ 44 cột của các dòng train từ CSV gốc, kiểm tra missing/unique/ID/time/coordinate/count trực tiếp trên train. |
| 1.3–1.8 EDA | Bỏ sample_all44 và mọi bảng *_full.csv. Mẫu biểu đồ lấy từ train, tối đa 200.000 dòng. Bảng thống kê/đếm/tỷ lệ tính trên toàn bộ train. |
| Dictionary | Chỉ giữ column/group/description/unit_or_encoding từ phần mô tả tĩnh. dtype/unique/missing tính lại trên train. |
| Category | Không lập bảng category/nhãn trên validation hoặc test, không audit tháng của holdout. Chỉ phân tích thời gian trong train. |
| 1.9 Kiểm định | Giữ công thức chi-square, Holm, Cramér V; dùng toàn bộ train. |
| 1.10 Features | Giữ 59 biến/công thức cũ và fit scaler/median/vocabulary trên train. Sau EDA mới áp dụng quy tắc đã cố định cho các dòng held-out; không học thống kê từ held-out. |
| 1.11 Handoff | Chỉ kiểm tra schema, thứ tự biến, số dòng. Không có fraud/fraud_pct của test trong phần này. |

## Điểm cần điều chỉnh so với gợi ý trong ảnh

- Đã bỏ cờ vận hành test_audit_disclosed khỏi settings và bỏ audit test khỏi luồng EDA mới.
- Tuy nhiên, các phiên bản trước thực tế đã xem nhãn test. Vì vậy README vẫn khai báo lịch sử này; không tuyên bố test cũ là chưa từng được xem.
- Handoff_Inputs cũ chỉ có các trường phục vụ mô hình, không đủ 44 cột. Bản sửa dùng CSV gốc, không dùng mẫu toàn bộ timeline làm fallback.

## Phần mô hình được giữ đến đâu

Có 16/21 cell code mô hình giữ nguyên từng ký tự so với snapshot Drive mới nhất. Các lệnh fit, tham số, helper evaluate, cách chia fold và cách chọn ngưỡng không bị viết lại.

Năm cell có thay đổi:

- Cell LightGBM giữ nguyên cấu hình mới được nhóm chốt (31 lá, min leaf 300, neg bagging 0.5, L2=1); chỉ nối evaluate để tạo thr_gb và giải phóng X_tr/y_tr sau khi train.
- Một cell dọn RAM bỏ việc xóa X_va/y_va sớm, vì cell test gốc sẽ xóa chúng. Đây là sửa lỗi xóa cùng biến hai lần trong bản trên Drive.
- Ba cell đánh giá/biểu đồ test chỉ được bọc trong if RUN_FINAL_TEST; nội dung tính toán bên trong giữ nguyên.

Số cell dưới đây là vị trí trong snapshot Drive trước khi sửa, đếm từ 1 (bao gồm Markdown).

### Cell cũ 123

```diff
--- trước
+++ sau
@@ -74,2 +74,6 @@
 print(f"Number of features: {len(FEATURES)}")
 print(f"Validation predictions: {len(val_scores_gb):,}")
+
+thr_gb, _ = evaluate('LightGBM · Validation', y_va, val_scores_gb)
+del X_tr, y_tr
+gc.collect()
```

### Cell cũ 141

```diff
--- trước
+++ sau
@@ -1,2 +1,2 @@
-del X_train, y_train, X_va, y_va, X_cv, y_cv, y_cv_array, val_scores
+del X_train, y_train, X_cv, y_cv, y_cv_array, val_scores
 gc.collect()
```

### Cell cũ 142

```diff
--- trước
+++ sau
@@ -1,8 +1,9 @@
-del X_va, y_va, dval, dtrain; gc.collect()
-X_te, y_te = load('test')
-print(f'Test: {X_te.shape[0]:,} dòng | fraud {y_te.mean():.2%}')
-
-p_lr = lr.predict_proba(X_te)[:, 1]
-p_gb = booster.predict(X_te, num_iteration=booster.best_iteration)
-evaluate('LR · Test', y_te, p_lr, thr_lr)
-evaluate('LightGBM · Test', y_te, p_gb, thr_gb);
+if RUN_FINAL_TEST:
+    del X_va, y_va, dval, dtrain; gc.collect()
+    X_te, y_te = load('test')
+    print(f'Test: {X_te.shape[0]:,} dòng | fraud {y_te.mean():.2%}')
+    
+    p_lr = lr.predict_proba(X_te)[:, 1]
+    p_gb = booster.predict(X_te, num_iteration=booster.best_iteration)
+    evaluate('LR · Test', y_te, p_lr, thr_lr)
+    evaluate('LightGBM · Test', y_te, p_gb, thr_gb);
```

### Cell cũ 144

```diff
--- trước
+++ sau
@@ -1,12 +1,13 @@
-# Precision-recall
-colors = {'LR · Test': '#4C78A8', 'LightGBM · Test': '#F58518'}
-
-fig, ax = plt.subplots(figsize=(11, 5))
-for name, p in [('LR · Test', p_lr), ('LightGBM · Test', p_gb)]:
-    pr, rc, _ = precision_recall_curve(y_te, p); r = RESULTS[name]
-    ax.plot(rc, pr, lw=2, color=colors[name], label=f"{name.split(' · ')[0]} (PR-AUC {r['PR-AUC']:.3f})")
-    ax.scatter(r['Recall'], r['Precision'], s=70, color=colors[name], edgecolor='k', zorder=3)
-ax.axhline(y_te.mean(), ls='--', c='gray', lw=1, label=f'Ngẫu nhiên ({y_te.mean():.2%})')
-ax.set(title='Precision–Recall (Test)  |  ● = ngưỡng chọn trên Validation', xlabel='Recall',
-       ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
-ax.legend(loc='upper right'); plt.tight_layout(); plt.show()
+if RUN_FINAL_TEST:
+    # Precision-recall
+    colors = {'LR · Test': '#4C78A8', 'LightGBM · Test': '#F58518'}
+    
+    fig, ax = plt.subplots(figsize=(11, 5))
+    for name, p in [('LR · Test', p_lr), ('LightGBM · Test', p_gb)]:
+        pr, rc, _ = precision_recall_curve(y_te, p); r = RESULTS[name]
+        ax.plot(rc, pr, lw=2, color=colors[name], label=f"{name.split(' · ')[0]} (PR-AUC {r['PR-AUC']:.3f})")
+        ax.scatter(r['Recall'], r['Precision'], s=70, color=colors[name], edgecolor='k', zorder=3)
+    ax.axhline(y_te.mean(), ls='--', c='gray', lw=1, label=f'Ngẫu nhiên ({y_te.mean():.2%})')
+    ax.set(title='Precision–Recall (Test)  |  ● = ngưỡng chọn trên Validation', xlabel='Recall',
+           ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
+    ax.legend(loc='upper right'); plt.tight_layout(); plt.show()
```

### Cell cũ 145

```diff
--- trước
+++ sau
@@ -1,13 +1,14 @@
-# Confusion matrix
-r = RESULTS['LightGBM · Test']
-cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]]); cmn = cm / cm.sum(1, keepdims=True)
-
-fig, ax = plt.subplots(figsize=(7, 5))
-ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1); ax.grid(False)
-for i in range(2):
-    for j in range(2):
-        ax.text(j, i, f'{cm[i, j]:,.0f}\n({cmn[i, j]:.1%})', ha='center', va='center',
-                color='white' if cmn[i, j] > .5 else 'black', fontsize=12)
-ax.set(title='Ma trận nhầm lẫn — LightGBM (Test)', xticks=[0, 1], yticks=[0, 1],
-       xticklabels=['Dự đoán thường', 'Dự đoán fraud'], yticklabels=['Thực tế thường', 'Thực tế fraud'])
-plt.tight_layout(); plt.show()
+if RUN_FINAL_TEST:
+    # Confusion matrix
+    r = RESULTS['LightGBM · Test']
+    cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]]); cmn = cm / cm.sum(1, keepdims=True)
+    
+    fig, ax = plt.subplots(figsize=(7, 5))
+    ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1); ax.grid(False)
+    for i in range(2):
+        for j in range(2):
+            ax.text(j, i, f'{cm[i, j]:,.0f}\n({cmn[i, j]:.1%})', ha='center', va='center',
+                    color='white' if cmn[i, j] > .5 else 'black', fontsize=12)
+    ax.set(title='Ma trận nhầm lẫn — LightGBM (Test)', xticks=[0, 1], yticks=[0, 1],
+           xticklabels=['Dự đoán thường', 'Dự đoán fraud'], yticklabels=['Thực tế thường', 'Thực tế fraud'])
+    plt.tight_layout(); plt.show()
```

## Kiểm tra đã chạy

- Chạy toàn bộ notebook bằng CSV thật 8.580.255 dòng, không lỗi cell; chế độ review không đánh giá test thật.
- Đối chiếu source_row và datetime của cả ba split với phiên bản cũ, chỉ đọc metadata, không đọc nhãn held-out để đối chiếu.
- Xác nhận mọi count trong numeric summary của train đều là 5.491.362.
- Xác nhận scaler fit đúng 5.491.362 dòng và schema có 59 biến.
- Kiểm tra tĩnh EDA không đọc file thống kê toàn bộ dữ liệu và không dùng biến held-out.
- Kiểm tra cổng final-test bằng dữ liệu tổng hợp nhỏ: khi False không gọi load(test), khi True gọi đúng nhánh đánh giá. Không dùng nhãn test thật trong kiểm tra này.
- Ghi nhận môi trường chạy và các phiên bản thư viện trong requirements-tested.txt.

## Phần nhóm vẫn cần quyết định

- Điền nguồn/license dataset và thông tin thành viên.
- Đọc hạn chế CV: scaler/vocabulary vẫn fit final train trước khi tạo fold nội bộ, giống phương pháp nhóm đã nộp. Chưa tự viết lại CV để tránh thay đổi ngoài phạm vi.
- Sau khi nhóm chốt mô hình mới bật RUN_FINAL_TEST=True để chấm cuối; không chọn lại mô hình theo điểm test.
- Kiểm tra xong mới tạo repo/commit/push. Bản chuẩn bị này chưa có Git remote hay lịch sử commit.