-- Read-only live Supabase schema inventory.

with target_tables(table_name) as (
    values
        ('sources'),
        ('chunks'),
        ('concepts'),
        ('chunk_concepts')
),
table_inventory as (
    select
        t.table_name,
        (c.oid is not null) as exists,
        coalesce(c.relrowsecurity, false) as rls_enabled,
        coalesce(c.relforcerowsecurity, false) as rls_forced
    from target_tables t
    left join pg_namespace n
        on n.nspname = 'public'
    left join pg_class c
        on c.relnamespace = n.oid
       and c.relname = t.table_name
       and c.relkind in ('r', 'p')
),
column_inventory as (
    select
        c.relname as table_name,
        a.attnum as ordinal_position,
        a.attname as column_name,
        pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type,
        not a.attnotnull as is_nullable,
        pg_get_expr(ad.adbin, ad.adrelid) as column_default
    from pg_class c
    join pg_namespace n
        on n.oid = c.relnamespace
    join pg_attribute a
        on a.attrelid = c.oid
       and a.attnum > 0
       and not a.attisdropped
    left join pg_attrdef ad
        on ad.adrelid = c.oid
       and ad.adnum = a.attnum
    where n.nspname = 'public'
      and c.relname in (
          'sources',
          'chunks',
          'concepts',
          'chunk_concepts'
      )
      and c.relkind in ('r', 'p')
),
constraint_inventory as (
    select
        c.relname as table_name,
        con.conname as constraint_name,
        case con.contype
            when 'p' then 'PRIMARY KEY'
            when 'f' then 'FOREIGN KEY'
            when 'u' then 'UNIQUE'
            when 'c' then 'CHECK'
            when 'x' then 'EXCLUSION'
            else con.contype::text
        end as constraint_type,
        pg_get_constraintdef(con.oid, true) as definition
    from pg_constraint con
    join pg_class c
        on c.oid = con.conrelid
    join pg_namespace n
        on n.oid = c.relnamespace
    where n.nspname = 'public'
      and c.relname in (
          'sources',
          'chunks',
          'concepts',
          'chunk_concepts'
      )
),
index_inventory as (
    select
        tablename as table_name,
        indexname as index_name,
        indexdef as definition
    from pg_indexes
    where schemaname = 'public'
      and tablename in (
          'sources',
          'chunks',
          'concepts',
          'chunk_concepts'
      )
),
vector_extension as (
    select
        extname,
        extversion
    from pg_extension
    where extname = 'vector'
),
schema_migrations as (
    select
        n.nspname as schema_name,
        c.relname as relation_name,
        c.relkind::text as relation_kind
    from pg_class c
    join pg_namespace n
        on n.oid = c.relnamespace
    where n.nspname in ('public', 'supabase_migrations')
      and (
          c.relname ilike '%migration%'
          or c.relname ilike '%schema%'
      )
)
select jsonb_pretty(
    jsonb_build_object(
        'audit_name',
        'wth_stage0_phase1_production_schema_inventory',
        'schema',
        'public',
        'read_only',
        true,
        'vector_extension',
        coalesce(
            (
                select jsonb_agg(
                    jsonb_build_object(
                        'name', extname,
                        'version', extversion
                    )
                    order by extname
                )
                from vector_extension
            ),
            '[]'::jsonb
        ),
        'tables',
        coalesce(
            (
                select jsonb_agg(
                    jsonb_build_object(
                        'table', ti.table_name,
                        'exists', ti.exists,
                        'rls_enabled', ti.rls_enabled,
                        'rls_forced', ti.rls_forced,
                        'columns',
                        coalesce(
                            (
                                select jsonb_agg(
                                    jsonb_build_object(
                                        'ordinal_position', ci.ordinal_position,
                                        'name', ci.column_name,
                                        'type', ci.data_type,
                                        'nullable', ci.is_nullable,
                                        'default', ci.column_default
                                    )
                                    order by ci.ordinal_position
                                )
                                from column_inventory ci
                                where ci.table_name = ti.table_name
                            ),
                            '[]'::jsonb
                        ),
                        'constraints',
                        coalesce(
                            (
                                select jsonb_agg(
                                    jsonb_build_object(
                                        'name', coi.constraint_name,
                                        'type', coi.constraint_type,
                                        'definition', coi.definition
                                    )
                                    order by
                                        coi.constraint_type,
                                        coi.constraint_name
                                )
                                from constraint_inventory coi
                                where coi.table_name = ti.table_name
                            ),
                            '[]'::jsonb
                        ),
                        'indexes',
                        coalesce(
                            (
                                select jsonb_agg(
                                    jsonb_build_object(
                                        'name', ii.index_name,
                                        'definition', ii.definition
                                    )
                                    order by ii.index_name
                                )
                                from index_inventory ii
                                where ii.table_name = ti.table_name
                            ),
                            '[]'::jsonb
                        )
                    )
                    order by ti.table_name
                )
                from table_inventory ti
            ),
            '[]'::jsonb
        ),
        'migration_related_relations',
        coalesce(
            (
                select jsonb_agg(
                    jsonb_build_object(
                        'schema', schema_name,
                        'name', relation_name,
                        'kind', relation_kind
                    )
                    order by schema_name, relation_name
                )
                from schema_migrations
            ),
            '[]'::jsonb
        )
    )
) as stage0_schema_inventory;
